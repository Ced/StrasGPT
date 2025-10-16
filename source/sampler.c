#include "sampler.h"
#include "options.h"
#include "transformer.h"
#include "util.h"
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
    UTIL_DIE("failed to malloc for tokenizer_t");
  }
  sampler->vocabulary_len = transformer->config->vocabulary_len;
  sampler->temperature = options->temperature;
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
    UTIL_DIE("failed to malloc for tokenizer_t");
  }
  return sampler;
}

// Free a sampler structure
void sampler_free(sampler_t* sampler) {
  if (!sampler) {
    return;
  }
  free(sampler->probindex);
  free(sampler);
}

// Print a sampler structure
void sampler_print(FILE* f, const sampler_t* sampler) {
  if (!sampler) {
    fprintf(f, "sampler: NULL\n");
    return;
  }
  fprintf(f, "Sampler:\n");
  fprintf(f, "- vocabulary_len: %zu\n", sampler->vocabulary_len);
  fprintf(f, "- temperature:    %f\n", sampler->temperature);
  fprintf(f, "- top_p:          %f\n", sampler->top_p);
  fprintf(f, "- rng_state:      %llu\n", sampler->rng_state);
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

// Top-p sampling (or "nucleus sampling") samples from the smallest set of
// tokens that exceed probability top_p. This way we never sample tokens that
// have very low probabilities and are less likely to go "off the rails".
// coin is a random number in [0, 1[, usually from random_f32()
static size_t sample_top_p(
    size_t len,
    float* probability,
    float top_p,
    sampler_probability_index_t* probindex,
    float coin
) {
  size_t len0 = 0;
  // Quicksort indices in descending order of probabilities
  // values smaller than (1 - top_p) / (len - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - top_p) / (len - 1);
  for (size_t i = 0; i < len; i++) {
    if (probability[i] >= cutoff) {
      probindex[len0].index = i;
      probindex[len0].probability = probability[i];
      len0++;
    }
  }
  qsort(probindex, len0, sizeof(*probindex), compare);

  // Truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  size_t last_idx = len0 - 1; // If rounding errors consider all elements
  for (size_t i = 0; i < len0; i++) {
    cumulative_prob += probindex[i].probability;
    if (cumulative_prob > top_p) {
      last_idx = i;
      break; // We've exceeded top_p by including last_idx
    }
  }

  // Sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (size_t i = 0; i <= last_idx; i++) {
    cdf += probindex[i].probability;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // In case of rounding errors
}

// Sample the next token given the logits and some hyperparameters
size_t sampler_sample(sampler_t* sampler, float* logits) {
  // Sample the token given the logits and some hyperparameters
  size_t next;
  if (sampler->temperature == 0.0f) {
    // Greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocabulary_len);
  } else {
    // Apply the temperature to the logits
    for (size_t q = 0; q < sampler->vocabulary_len; q++) {
      logits[q] /= sampler->temperature;
    }
    // Apply softmax to the logits to get the probabilities for next token
    softmax(sampler->vocabulary_len, logits);
    // Flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // We sample from this distribution to get the next token
    if (sampler->top_p <= 0 || sampler->top_p >= 1) {
      // Simply sample from the predicted probability distribution
      next = sample_mult(sampler->vocabulary_len, logits, coin);
    } else {
      // Top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_top_p(
          sampler->vocabulary_len,
          logits,
          sampler->top_p,
          sampler->probindex,
          coin
      );
    }
  }
  return next;
}
