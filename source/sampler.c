#include "sampler.h"
#include "options.h"
#include "transformer.h"
#include "util.h"
#ifdef DEBUG
#include "tokenizer.h"
#endif
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Build a sampler from options_t and transformer_t configuration
sampler_t* sampler_build(options_t* options, transformer_t* transformer) {
  if (!options || !transformer) {
    UTIL_DIE("options or transformer is NULL");
  }

  sampler_t* sampler = calloc(1, sizeof(sampler_t));
  if (!sampler) {
    UTIL_DIE("failed to malloc for sampler_t");
  }
  #ifdef DEBUG
  sampler->tokenizer = NULL;
  #endif
  sampler->vocabulary_len = transformer->config->vocabulary_len;
  sampler->temperature = options->temperature;
  sampler->top_k = options->top_k;
  sampler->top_p = options->top_p;
  sampler->rng_state = options->seed;
  if (options->seed <= 0) {
    sampler->rng_state = (unsigned long long)time(NULL);
  } else {
    sampler->rng_state = options->seed;
  }

  // Buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex =
      calloc(sampler->vocabulary_len, sizeof(*sampler->probindex));
  if (!sampler->probindex) {
    UTIL_DIE("failed to malloc for probindex");
  }

  // Token presence buffer for presence penalty
  sampler->token_presence = calloc(sampler->vocabulary_len, sizeof(bool));
  if (!sampler->token_presence) {
    UTIL_DIE("failed to malloc for token_presence");
  }
  return sampler;
}

// Free a sampler structure
void sampler_free(sampler_t* sampler) {
  if (!sampler) {
    return;
  }
  free(sampler->probindex);
  free(sampler->token_presence);
  free(sampler);
}

// Print a sampler structure
void sampler_print(FILE* f, const sampler_t* sampler) {
  if (!sampler) {
    fprintf(f, "sampler: NULL\n");
    return;
  }
  fprintf(f, "Sampler:\n");
  fprintf(f, "- vocabulary_len:   %zu\n", sampler->vocabulary_len);
  fprintf(f, "- temperature:      %f\n", sampler->temperature);
  fprintf(f, "- presence_penalty: %f\n", sampler->presence_penalty);
  fprintf(f, "- top_k:            %zu\n", sampler->top_k);
  fprintf(f, "- top_p:            %f\n", sampler->top_p);
  fprintf(f, "- rng_state:        %llu\n", sampler->rng_state);
}

// Xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
static unsigned int random_u32(unsigned long long* state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// Random float32 in [0,1)
static float random_f32(unsigned long long* state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

// Return the index that has the highest probability
static size_t sample_argmax(float* probability, size_t n) {
  size_t max_i = 0;
  float max_p = probability[0];
  for (size_t i = 1; i < n; i++) {
    if (probability[i] > max_p) {
      max_i = i;
      max_p = probability[i];
    }
  }
  return max_i;
}

// Apply softmax to an array of floats in place
static void softmax(size_t len, float* x) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (size_t i = 1; i < len; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (size_t i = 0; i < len; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (size_t i = 0; i < len; i++) {
    x[i] /= sum;
  }
}

// Sample index from probabilities (they must sum to 1!)
// coin is a random number in [0, 1[, usually from random_f32()
static size_t sample_mult(size_t len, float* probability, float coin) {
  float cdf = 0.0f;
  for (size_t i = 0; i < len; i++) {
    cdf += probability[i];
    if (coin < cdf) {
      return i;
    }
  }
  return len - 1; // In case of rounding errors
}

// Renormalize probabilities to sum to 1 in place
static void renormalize(size_t n, float* p) {
  double sum = 0.0;

  // Accumulate in double for numerical stability
  for (size_t i = 0; i < n; i++) {
    sum += (double)p[i];
  }

  // Fallback: uniform if everything was masked out
  if (sum <= 0.0) {
    const float u = 1.0f / (float)n;
    for (size_t i = 0; i < n; i++) {
      p[i] = u;
    }
    return;
  }

  const float inv_sum = (float)(1.0 / sum);
  for (size_t i = 0; i < n; i++) {
    p[i] *= inv_sum;
  }
}

// Comparison function for qsort: descending order of probability
static int compare(const void* a, const void* b) {
  sampler_probability_index_t* a_ = (sampler_probability_index_t*)a;
  sampler_probability_index_t* b_ = (sampler_probability_index_t*)b;
  if (a_->probability > b_->probability)
    return -1;
  if (a_->probability < b_->probability)
    return 1;
  return 0;
}

// Top-k filtering: keeps only the top-k most likely tokens
static void filter_top_k(
    size_t len,
    float* probability,
    size_t k,
    sampler_probability_index_t* probindex
) {
  // Copy all probabilities with their indices
  for (size_t i = 0; i < len; i++) {
    probindex[i].index = i;
    probindex[i].probability = probability[i];
  }

  // Sort by probability (descending order)
  qsort(probindex, len, sizeof(*probindex), compare);

  // Limit to top-k
  size_t k_actual = k < len ? k : len;

  // Zero out probabilities of tokens outside top-k
  for (size_t i = k_actual; i < len; i++) {
    probability[probindex[i].index] = 0.0f;
  }
}

// Top-p filtering (or "nucleus sampling") filters to the smallest set of
// tokens that exceed probability top_p.
static void filter_top_p(
    size_t len,
    float* probability,
    float top_p,
    sampler_probability_index_t* probindex
) {
  // Copy all probabilities with their indices
  for (size_t i = 0; i < len; i++) {
    probindex[i].index = i;
    probindex[i].probability = probability[i];
  }

  // Sort by probability (descending order)
  qsort(probindex, len, sizeof(*probindex), compare);

  // Truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  size_t last_idx = len - 1; // If rounding errors consider all elements
  for (size_t i = 0; i < len; i++) {
    cumulative_prob += probindex[i].probability;
    if (cumulative_prob >= top_p) {
      last_idx = i;
      break; // We've exceeded top_p by including last_idx
    }
  }

  // Zero out probabilities of tokens outside top-p
  for (size_t i = last_idx + 1; i < len; i++) {
    probability[probindex[i].index] = 0.0f;
  }
}

// Sample the next token given the logits and some hyperparameters
// Also provide the previous token to apply presence penalty
size_t sampler_sample(sampler_t* sampler, float* logits, int token) {
  // Mark the previous token as present (if valid)
  if (token >= 0 && (size_t)token < sampler->vocabulary_len) {
    sampler->token_presence[token] = true;
  }

  // Apply presence penalty to logits before any processing
  if (sampler->presence_penalty != 0.0f) {
    for (size_t i = 0; i < sampler->vocabulary_len; i++) {
      if (sampler->token_presence[i]) {
        logits[i] -= sampler->presence_penalty;
      }
    }
  }

  // Sample the token given the logits and some hyperparameters
  size_t next;
  if (sampler->temperature == 0.0f) {
    // Greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocabulary_len);
  } else {
    #ifdef DEBUG
    // Print logits summary
    fprintf(stderr, "Transformer output:\n");
    util_matrix_summary("-    logits", 1, sampler->vocabulary_len, 3, logits);
    #endif

    // Apply the temperature to the logits
    for (size_t q = 0; q < sampler->vocabulary_len; q++) {
      logits[q] /= sampler->temperature;
    }
    // Apply softmax to the logits to get the probabilities for next token
    softmax(sampler->vocabulary_len, logits);

    #ifdef DEBUG
    // Print top tokens with highest probability
    sampler_probability_index_t top[SAMPLER_DEBUG_TOP_TOKENS];
    for (size_t i = 0; i < SAMPLER_DEBUG_TOP_TOKENS; i++) {
      top[i].index = 0;
      top[i].probability = -1.0f;
    }
    // Find top by simple linear search
    for (size_t i = 0; i < sampler->vocabulary_len; i++) {
      float prob = logits[i];
      // Check if this probability is in top
      for (size_t j = 0; j < SAMPLER_DEBUG_TOP_TOKENS; j++) {
        if (prob > top[j].probability) {
          // Shift lower probabilities down
          for (size_t k = SAMPLER_DEBUG_TOP_TOKENS - 1; k > j; k--) {
            top[k] = top[k - 1];
          }
          top[j].index = i;
          top[j].probability = prob;
          break;
        }
      }
    }
    fprintf(stderr, "\nTop%d tokens:\n", SAMPLER_DEBUG_TOP_TOKENS);
    for (size_t i = 0; i < SAMPLER_DEBUG_TOP_TOKENS; i++) {
      fprintf(
          stderr,
          "- %2zu token=%6zu probability=%7.4f",
          i + 1,
          top[i].index,
          top[i].probability
      );
      if (sampler->tokenizer) {
        fprintf(
            stderr,
            " text=\"%s\"",
            tokenizer_decode(sampler->tokenizer, top[i].index)
        );
      }
      fprintf(stderr, "\n");
    }
    #endif

    // Flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);

    // We sample from this distribution to get the next token
    // Apply filters in sequence: top-k first, then top-p
    if (sampler->top_k > 0) {
      // First apply top-k filtering
      filter_top_k(
          sampler->vocabulary_len, logits, sampler->top_k, sampler->probindex
      );
    }

    if (sampler->top_p > 0 && sampler->top_p <= 1) {
      filter_top_p(
          sampler->vocabulary_len, logits, sampler->top_p, sampler->probindex
      );
    }

    // Renormalize the filtered probabilities
    renormalize(sampler->vocabulary_len, logits);

    // Sample from the normalized distribution
    next = sample_mult(sampler->vocabulary_len, logits, coin);
  }

  return next;
}
