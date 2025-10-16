#include "safetensors.h"
#include "transformer.h"
#include "util.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/mman.h>   // for mmap(), munmap(), PROT_*, MAP_* constants
#include <sys/types.h>  // for size_t, off_t
#include <fcntl.h>      // for open() and O_* flags
#include <unistd.h>     // for close()

// Return true if the output weight is aliased to the embedding weight
// If the output weight is not found, we assume it's aliased
static bool aliased_out_weight(const safetensors_t* safetensors) {
  for (size_t i = 0; i < safetensors->tensor_count; i++) {
    const safetensors_tensor_t* t = &safetensors->tensor[i];
    if (strcmp(t->name, SAFETENSORS_PATTERN_OUT_WEIGHT) == 0) {
      return false; // Found the output weight, not aliased
    }
  }
  return true;
}

// Create a transformer_configuration_t structure from a safetensors_t
static transformer_configuration_t* configuration_from_safetensors(
    safetensors_t* safetensors
) {
  transformer_configuration_t* config = malloc(sizeof(*config));
  if (config == NULL) {
    UTIL_DIE("failed to malloc for transformer_configuration_t");
  }
  config->embedding_dim = safetensors->embedding_dim;
  config->hidden_dim = safetensors->hidden_dim;
  config->layer_count = safetensors->layer_count;
  config->q_head_count = safetensors->q_head_count;
  config->kv_head_count = safetensors->kv_head_count;
  config->vocabulary_len = safetensors->vocabulary_len;
  config->context_len = safetensors->context_len;
  config->aliased_out_weight = aliased_out_weight(safetensors);
  return config;
}

// Create a transformer_state_t structure from a safetensors_t
static transformer_state_t* state_from_safetensors(safetensors_t* t) {
  transformer_state_t* s = calloc(1, sizeof(*s));
  if (s == NULL) {
    UTIL_DIE("failed to malloc for transformer_state_t");
  }

  size_t chunk_max_len = TRANSFORMER_CHUNK_MAX_LEN;
  size_t head_dim = t->embedding_dim / t->q_head_count;
  size_t kv_dim = head_dim * t->kv_head_count;
  size_t embedding_len = chunk_max_len * t->embedding_dim;
  size_t hidden_len = chunk_max_len * t->hidden_dim;
  size_t score_len = t->q_head_count * chunk_max_len * t->context_len;
  size_t cache_len = t->context_len * t->layer_count * kv_dim;
  size_t logits_len = chunk_max_len * t->vocabulary_len;
  size_t rope_len = t->context_len * head_dim;

  s->embedding = calloc(embedding_len, sizeof(*s->embedding));
  s->mha_norm = calloc(embedding_len, sizeof(*s->mha_norm));
  s->mha_q = calloc(embedding_len, sizeof(*s->mha_q));
  s->mha_score = calloc(score_len, sizeof(*s->mha_score));
  s->mha_att = calloc(embedding_len, sizeof(*s->mha_att));
  s->mha_out = calloc(embedding_len, sizeof(*s->mha_out));
  s->ffn_norm = calloc(embedding_len, sizeof(*s->ffn_norm));
  s->ffn_fc = calloc(hidden_len, sizeof(*s->ffn_fc));
  s->ffn_up = calloc(hidden_len, sizeof(*s->ffn_up));
  s->ffn_out = calloc(embedding_len, sizeof(*s->ffn_out));
  s->logits = calloc(logits_len, sizeof(float));
  s->k_cache = calloc(cache_len, sizeof(*s->k_cache));
  s->v_cache = calloc(cache_len, sizeof(*s->v_cache));
  s->rope_cos_sin = calloc(rope_len, sizeof(*s->rope_cos_sin));

  // Ensure all mallocs went fine
  if (!s->embedding ||
      !s->mha_norm ||
      !s->mha_q ||
      !s->mha_score ||
      !s->mha_att ||
      !s->mha_out ||
      !s->ffn_norm ||
      !s->ffn_fc ||
      !s->ffn_up ||
      !s->ffn_out ||
      !s->logits ||
      !s->k_cache ||
      !s->v_cache ||
      !s->rope_cos_sin) {
    UTIL_DIE("failed to malloc for activations");
  }

  // Initialize RoPE cosine and sine values
  for (size_t i = 0; i < t->context_len; i++) {
    for (size_t j = 0; j < head_dim; j += 2) {
      float freq = 1.0f / powf(500000.0f, j / (float)head_dim);
      float val = i * freq;
      s->rope_cos_sin[i * head_dim + j] = cosf(val);
      s->rope_cos_sin[i * head_dim + j + 1] = sinf(val);
    }
  }

  return s;
}

// Permute Hugging Face weights (grouped RoPE layout) to Meta layout
// (interleaved). The permutation matches the Python reference:
// def permute_original(w, n_heads, dim1=dim, dim2=dim):
//   return (
//     w.view(dim1, dim2)
//     .reshape(n_heads, dim1 // n_heads // 2, 2, dim2)
//     .transpose(1, 2)
//     .reshape(dim1, dim2)
//   )
static void permute_hf_to_meta(
    size_t row_count, size_t col_count, size_t head_count, uint16_t* w
) {
  size_t head_dim = row_count / head_count;
  size_t half_head = head_dim / 2;

  uint16_t* buf = malloc(row_count * col_count * sizeof(uint16_t));
  if (!buf) {
    UTIL_DIE("failed to malloc for buffer");
  }

  for (size_t h = 0; h < head_count; h++) {
    // Pointers to start of this head in source and destination
    const uint16_t* src_head = w + h * head_dim * col_count;
    uint16_t* dst_head = buf + h * head_dim * col_count;

    // Reshape: [half_head, 2, col_count] -> transpose(0,1)
    for (size_t j = 0; j < half_head; j++) {
      for (size_t k = 0; k < 2; k++) {
        // In HF layout:
        // - first half_head rows: "real" parts
        // - second half_head rows: "imag" parts
        // After transpose, we interleave them.
        const uint16_t* src = src_head + (k * half_head + j) * col_count;
        uint16_t* dst = dst_head + (2 * j + k) * col_count;
        memcpy(dst, src, col_count * sizeof(uint16_t));
      }
    }
  }

  memcpy(w, buf, row_count * col_count * sizeof(uint16_t));
  free(buf);
}

// Match an unsigned integer at the start of *string, update *string to point
// after the integer, and set *index to the parsed value. Returns true if an
// integer was found, false otherwise
static bool match_index(const char** string, size_t* index) {
  const char* s = *string;

  if (!isdigit((unsigned char)*s)) {
    return false;
  }

  size_t i = 0;
  while (isdigit((unsigned char)*s)) {
    i = i * 10 + (*s - '0');
    s++;
  }

  *index = i;
  *string = s;
  return true;
}

// Match a name against a pattern with possiby a single %d for an integer index
// If matched, set *index to the parsed index (or 0 if no %d) and return true
// If not matched, return false and leave *index unchanged
static bool match_name(const char* name, const char* pattern, size_t* index) {
  int index_count = 0;
  const char* n = name;
  const char* p = pattern;

  while (*p) {
    // Scan an index from the name if we see a %d in the pattern
    if (p[0] == '%' && p[1] == 'd') {
      if (index_count >= 1) {
        UTIL_DIE("only one index supported yet");
      }

      size_t index_value;
      if (!match_index(&n, &index_value)) {
        return false;
      }
      *index = index_value;
      index_count++;
      p += 2;
      continue;
    }
    // Otherwise, characters must match
    if (*n != *p) {
      return false;
    }
    n++;
    p++;
  }
  // Pattern ended; name must also end
  if (*n != '\0') {
    return false;
  }
  return true;
}

// Load data from a file at a given offset into a storage buffer
static void load_data(
    const char* file, size_t offset, size_t size, void* storage
) {
  FILE* f = fopen(file, "rb");
  if (f == NULL) {
    UTIL_DIE("failed to open file");
  }
  if (fseek(f, offset, SEEK_SET) != 0) {
    UTIL_DIE("failed to seek in file");
  }
  size_t total = 0;
  while (total < size) {
    size_t n = fread((char*)storage + total, 1, size - total, f);
    if (n == 0) {
      UTIL_DIE("failed to read from file");
    }
    total += n;
  }
  if (fclose(f) != 0) {
    UTIL_DIE("failed to close file");
  }
}

static void load_embedding_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  (void)index; // Unused
  size_t len = safetensors->vocabulary_len * safetensors->embedding_dim;
  if (tensor->size != len * sizeof(*weights->embedding_weight)) {
    UTIL_DIE("unexpected size for embedding weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      weights->embedding_weight
  );
}

static void load_mha_norm_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t len = safetensors->embedding_dim;
  if (tensor->size != len * sizeof(*weights->mha_norm_weight)) {
    UTIL_DIE("unexpected size for mha norm weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->mha_norm_weight[index * len]
  );
}

static void load_mha_q_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t head_dim = safetensors->embedding_dim / safetensors->q_head_count;
  size_t qkv_weight_dim = head_dim * safetensors->embedding_dim;
  size_t len = safetensors->q_head_count * qkv_weight_dim;
  if (tensor->size != len * sizeof(*weights->mha_q_weight)) {
    UTIL_DIE("unexpected size for mha q weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->mha_q_weight[index * len]
  );

  // Permute the weight from Hugging Face layout to Meta layout
  permute_hf_to_meta(
      safetensors->q_head_count * head_dim,
      safetensors->embedding_dim,
      safetensors->q_head_count,
      &weights->mha_q_weight[index * len]
  );
}

static void load_mha_k_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t head_dim = safetensors->embedding_dim / safetensors->q_head_count;
  size_t qkv_weight_dim = head_dim * safetensors->embedding_dim;
  size_t len = safetensors->kv_head_count * qkv_weight_dim;
  if (tensor->size != len * sizeof(*weights->mha_k_weight)) {
    UTIL_DIE("unexpected size for mha k weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->mha_k_weight[index * len]
  );

  // Permute the weight from Hugging Face layout to Meta layout
  permute_hf_to_meta(
      safetensors->kv_head_count * head_dim,
      safetensors->embedding_dim,
      safetensors->kv_head_count,
      &weights->mha_k_weight[index * len]
  );
}

static void load_mha_v_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t head_dim = safetensors->embedding_dim / safetensors->q_head_count;
  size_t qkv_weight_dim = head_dim * safetensors->embedding_dim;
  size_t len = safetensors->kv_head_count * qkv_weight_dim;
  if (tensor->size != len * sizeof(*weights->mha_v_weight)) {
    UTIL_DIE("unexpected size for mha v weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->mha_v_weight[index * len]
  );
}

static void load_mha_out_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t len = safetensors->embedding_dim * safetensors->embedding_dim;
  if (tensor->size != len * sizeof(*weights->mha_out_weight)) {
    UTIL_DIE("unexpected size for mha out weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->mha_out_weight[index * len]
  );
}

static void load_ffn_norm_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t len = safetensors->embedding_dim;
  if (tensor->size != len * sizeof(*weights->ffn_norm_weight)) {
    UTIL_DIE("unexpected size for ffn norm weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->ffn_norm_weight[index * len]
  );
}

static void load_ffn_fc_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t len = safetensors->embedding_dim * safetensors->hidden_dim;
  if (tensor->size != len * sizeof(*weights->ffn_fc_weight)) {
    UTIL_DIE("unexpected size for ffn fc weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->ffn_fc_weight[index * len]
  );
}

static void load_ffn_up_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t len = safetensors->embedding_dim * safetensors->hidden_dim;
  if (tensor->size != len * sizeof(*weights->ffn_up_weight)) {
    UTIL_DIE("unexpected size for ffn up weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->ffn_up_weight[index * len]
  );
}

static void load_ffn_out_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  size_t len = safetensors->hidden_dim * safetensors->embedding_dim;
  if (tensor->size != len * sizeof(*weights->ffn_out_weight)) {
    UTIL_DIE("unexpected size for ffn out weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      &weights->ffn_out_weight[index * len]
  );
}

static void load_out_norm_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    size_t index,
    transformer_weights_t* weights
) {
  (void)index; // Unused
  size_t len = safetensors->embedding_dim;
  if (tensor->size != len * sizeof(*weights->out_norm_weight)) {
    UTIL_DIE("unexpected size for out norm weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      weights->out_norm_weight
  );
}

static void load_out_weight(
    const safetensors_t* safetensors,
    const safetensors_tensor_t* tensor,
    const size_t index,
    transformer_weights_t* weights
) {
  (void)index; // Unused
  size_t len = safetensors->vocabulary_len * safetensors->embedding_dim;
  if (tensor->size != len * sizeof(*weights->out_weight)) {
    UTIL_DIE("unexpected size for out weight");
  }

  load_data(
      safetensors->file[tensor->file],
      tensor->offset,
      tensor->size,
      weights->out_weight
  );
}

// Structure to map tensor name patterns to loading functions
typedef struct {
  const char* pattern; // e.g. "model.layers.%d.input_layernorm.weight"
  void (*loader)(      // Function to load the tensor with this pattern
      const safetensors_t* safetensors,
      const safetensors_tensor_t* tensor,
      size_t index,
      transformer_weights_t* weights
  );
} tensor_load_t;

// Load a tensor into the weights structure based on its name pattern
// Return true if the tensor was recognized and loaded, false otherwise
static bool tensor_load(
    const safetensors_tensor_t* tensor,
    const safetensors_t* safetensors,
    transformer_weights_t* weights
) {
  const tensor_load_t loading_table[] = {
      {SAFETENSORS_PATTERN_EMBEDDING_WEIGHT, load_embedding_weight},
      {SAFETENSORS_PATTERN_MHA_NORM_WEIGHT, load_mha_norm_weight},
      {SAFETENSORS_PATTERN_MHA_Q_WEIGHT, load_mha_q_weight},
      {SAFETENSORS_PATTERN_MHA_K_WEIGHT, load_mha_k_weight},
      {SAFETENSORS_PATTERN_MHA_V_WEIGHT, load_mha_v_weight},
      {SAFETENSORS_PATTERN_MHA_OUT_WEIGHT, load_mha_out_weight},
      {SAFETENSORS_PATTERN_FFN_NORM_WEIGHT, load_ffn_norm_weight},
      {SAFETENSORS_PATTERN_FFN_FC_WEIGHT, load_ffn_fc_weight},
      {SAFETENSORS_PATTERN_FFN_UP_WEIGHT, load_ffn_up_weight},
      {SAFETENSORS_PATTERN_FFN_OUT_WEIGHT, load_ffn_out_weight},
      {SAFETENSORS_PATTERN_OUT_NORM_WEIGHT, load_out_norm_weight},
      {SAFETENSORS_PATTERN_OUT_WEIGHT, load_out_weight}
  };
  size_t route_count = sizeof(loading_table) / sizeof(loading_table[0]);

  for (size_t i = 0; i < route_count; i++) {
    const char* name = tensor->name;
    size_t index = 0; // Will be set by match_name(), then unused

    if (match_name(name, loading_table[i].pattern, &index)) {
      loading_table[i].loader(safetensors, tensor, index, weights);
      return true;
    }
  }
  return false;
}

// Create a transformer_weights_t structure from a safetensors_t
static transformer_weights_t* weights_from_safetensors(safetensors_t* t) {
  transformer_weights_t* w = calloc(1, sizeof(*w));
  if (w == NULL) {
    UTIL_DIE("failed to malloc for transformer_weights_t");
  }

  size_t head_dim = t->embedding_dim / t->q_head_count;
  size_t qkv_weight_dim = head_dim * t->embedding_dim;

  size_t embedding_len = t->vocabulary_len * t->embedding_dim;
  size_t mha_norm_len = t->layer_count * t->embedding_dim;
  size_t mha_q_len = t->layer_count * t->q_head_count * qkv_weight_dim;
  size_t mha_kv_len = t->layer_count * t->kv_head_count * qkv_weight_dim;
  size_t mha_out_len = t->layer_count * t->embedding_dim * t->embedding_dim;
  size_t ffn_norm_len = t->layer_count * t->embedding_dim;
  size_t ffn_fc_len = t->layer_count * t->embedding_dim * t->hidden_dim;
  size_t ffn_up_len = t->layer_count * t->embedding_dim * t->hidden_dim;
  size_t ffn_out_len = t->layer_count * t->hidden_dim * t->embedding_dim;
  size_t out_norm_len = t->embedding_dim;
  size_t out_len = t->vocabulary_len * t->embedding_dim;

  w->embedding_weight = calloc(embedding_len, sizeof(*w->embedding_weight));
  w->mha_norm_weight = calloc(mha_norm_len, sizeof(*w->mha_norm_weight));
  w->mha_q_weight = calloc(mha_q_len, sizeof(*w->mha_q_weight));
  w->mha_k_weight = calloc(mha_kv_len, sizeof(*w->mha_k_weight));
  w->mha_v_weight = calloc(mha_kv_len, sizeof(*w->mha_v_weight));
  w->mha_out_weight = calloc(mha_out_len, sizeof(*w->mha_out_weight));
  w->ffn_norm_weight = calloc(ffn_norm_len, sizeof(*w->ffn_norm_weight));
  w->ffn_fc_weight = calloc(ffn_fc_len, sizeof(*w->ffn_fc_weight));
  w->ffn_up_weight = calloc(ffn_up_len, sizeof(*w->ffn_up_weight));
  w->ffn_out_weight = calloc(ffn_out_len, sizeof(*w->ffn_out_weight));
  w->out_norm_weight = calloc(out_norm_len, sizeof(*w->out_norm_weight));
  bool is_out_weigth_aliased = aliased_out_weight(t);
  if (is_out_weigth_aliased) {
    w->out_weight = w->embedding_weight;
  } else {
    w->out_weight = calloc(out_len, sizeof(*w->out_weight));
  }

  // Ensure all mallocs went fine
  if (!w->embedding_weight ||
      !w->mha_norm_weight ||
      !w->mha_q_weight ||
      !w->mha_k_weight ||
      !w->mha_v_weight ||
      !w->mha_out_weight ||
      !w->ffn_norm_weight ||
      !w->ffn_fc_weight ||
      !w->ffn_up_weight ||
      !w->ffn_out_weight ||
      !w->out_norm_weight ||
      (!w->out_weight && !is_out_weigth_aliased)) {
    UTIL_DIE("failed to malloc for weights");
  }

  // Load weights from safetensors
  for (size_t i = 0; i < t->tensor_count; i++) {
    if (!tensor_load(&t->tensor[i], t, w)) {
      fprintf(
          stderr, "[StrasGPT] Warning: unknown tensor %s\n", t->tensor[i].name
      );
    }
  }

  return w;
}

// Create a transformer_t structure from a safetensors_t
transformer_t* transformer_from_safetensors(safetensors_t* safetensors) {
  transformer_t* t = calloc(1, sizeof(*t));
  if (t == NULL) {
    UTIL_DIE("failed to malloc for transformer_t");
  }
  t->config = configuration_from_safetensors(safetensors);
  t->weights = weights_from_safetensors(safetensors);
  t->state = state_from_safetensors(safetensors);
  return t;
}

// Free a transformer_t structure
void transformer_free(transformer_t* transformer) {
  if (!transformer) {
    return;
  }

  transformer_configuration_t* c = transformer->config;
  transformer_weights_t* w = transformer->weights;
  transformer_state_t* s = transformer->state;

  free(w->embedding_weight);
  free(w->mha_norm_weight);
  free(w->mha_q_weight);
  free(w->mha_k_weight);
  free(w->mha_v_weight);
  free(w->mha_out_weight);
  free(w->ffn_norm_weight);
  free(w->ffn_fc_weight);
  free(w->ffn_up_weight);
  free(w->ffn_out_weight);
  free(w->out_norm_weight);
  if (!c->aliased_out_weight) {
    free(w->out_weight);
  }
  free(w);

  free(s->embedding);
  free(s->mha_norm);
  free(s->mha_q);
  free(s->mha_score);
  free(s->mha_att);
  free(s->mha_out);
  free(s->ffn_norm);
  free(s->ffn_fc);
  free(s->ffn_up);
  free(s->ffn_out);
  free(s->logits);
  free(s->k_cache);
  free(s->v_cache);
  free(s->rope_cos_sin);
  free(s);

  free(c);

  free(transformer);
}

// Print a summary of a transformer_t structure
void transformer_print(FILE* f, const transformer_t* transformer) {
  if (!transformer) {
    fprintf(f, "transformer: NULL\n");
    return;
  }

  transformer_configuration_t* c = transformer->config;

  fprintf(f, "Transformer:\n");
  fprintf(f, "- Configuration:\n");
  fprintf(f, "--- embedding_dim:      %zu\n", c->embedding_dim);
  fprintf(f, "--- hidden_dim:         %zu\n", c->hidden_dim);
  fprintf(f, "--- layer_count:        %zu\n", c->layer_count);
  fprintf(f, "--- q_head_count:       %zu\n", c->q_head_count);
  fprintf(f, "--- kv_head_count:      %zu\n", c->kv_head_count);
  fprintf(f, "--- vocabulary_len:     %zu\n", c->vocabulary_len);
  fprintf(f, "--- context_len:        %zu\n", c->context_len);
  char* aliased_out = c->aliased_out_weight ? "true" : "false";
  fprintf(f, "--- aliased_out_weight: %s\n", aliased_out);

  size_t head_dim = c->embedding_dim / c->q_head_count;
  size_t qkv_weight_dim = head_dim * c->embedding_dim;

  size_t embedding_len = c->vocabulary_len * c->embedding_dim;
  size_t mha_norm_len = c->layer_count * c->embedding_dim;
  size_t mha_q_len = c->layer_count * c->q_head_count * qkv_weight_dim;
  size_t mha_kv_len = c->layer_count * c->kv_head_count * qkv_weight_dim;
  size_t mha_out_len = c->layer_count * c->embedding_dim * c->embedding_dim;
  size_t ffn_norm_len = c->layer_count * c->embedding_dim;
  size_t ffn_fc_len = c->layer_count * c->embedding_dim * c->hidden_dim;
  size_t ffn_up_len = c->layer_count * c->embedding_dim * c->hidden_dim;
  size_t ffn_out_len = c->layer_count * c->hidden_dim * c->embedding_dim;
  size_t out_norm_len = c->embedding_dim;
  size_t out_len = c->vocabulary_len * c->embedding_dim;

  transformer_weights_t* w = transformer->weights;
  double gb = 1024 * 1024 * 1024;
  double embedding_gb = (embedding_len * sizeof(*w->embedding_weight)) / gb;
  double mha_norm_gb = (mha_norm_len * sizeof(*w->mha_norm_weight)) / gb;
  double mha_q_gb = (mha_q_len * sizeof(*w->mha_q_weight)) / gb;
  double mha_k_gb = (mha_kv_len * sizeof(*w->mha_k_weight)) / gb;
  double mha_v_gb = (mha_kv_len * sizeof(*w->mha_v_weight)) / gb;
  double mha_out_gb = (mha_out_len * sizeof(*w->mha_out_weight)) / gb;
  double ffn_norm_gb = (ffn_norm_len * sizeof(*w->ffn_norm_weight)) / gb;
  double ffn_fc_gb = (ffn_fc_len * sizeof(*w->ffn_fc_weight)) / gb;
  double ffn_up_gb = (ffn_up_len * sizeof(*w->ffn_up_weight)) / gb;
  double ffn_out_gb = (ffn_out_len * sizeof(*w->ffn_out_weight)) / gb;
  double out_norm_gb = (out_norm_len * sizeof(*w->out_norm_weight)) / gb;
  double out_gb =
      c->aliased_out_weight ? 0 : (out_len * sizeof(*w->out_weight)) / gb;

  double total_gb = embedding_gb + mha_norm_gb + mha_q_gb + mha_k_gb +
                    mha_v_gb + mha_out_gb + ffn_norm_gb + ffn_fc_gb +
                    ffn_up_gb + ffn_out_gb + out_norm_gb + out_gb;

  fprintf(f, "- Weights (%.2f GB):\n", total_gb);
  char s[SAFETENSORS_MAX_STRING];
  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "embedding", embedding_gb);
  util_matrix_summary(s, 1, embedding_len, 3, w->embedding_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "mha_norm", mha_norm_gb);
  util_matrix_summary(s, 1, mha_norm_len, 3, w->mha_norm_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "mha_q", mha_q_gb);
  util_matrix_summary(s, 1, mha_q_len, 3, w->mha_q_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "mha_k", mha_k_gb);
  util_matrix_summary(s, 1, mha_kv_len, 3, w->mha_k_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "mha_v", mha_v_gb);
  util_matrix_summary(s, 1, mha_kv_len, 3, w->mha_k_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "mha_out", mha_out_gb);
  util_matrix_summary(s, 1, mha_out_len, 3, w->mha_out_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "ffn_norm", ffn_norm_gb);
  util_matrix_summary(s, 1, ffn_norm_len, 3, w->ffn_norm_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "ffn_fc", ffn_fc_gb);
  util_matrix_summary(s, 1, ffn_fc_len, 3, w->ffn_fc_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "ffn_up", ffn_up_gb);
  util_matrix_summary(s, 1, ffn_up_len, 3, w->ffn_up_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "ffn_out", ffn_out_gb);
  util_matrix_summary(s, 1, ffn_out_len, 3, w->ffn_out_weight);

  snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "out_norm", out_norm_gb);
  util_matrix_summary(s, 1, out_norm_len, 3, w->out_norm_weight);

  if (c->aliased_out_weight) {
    fprintf(f, "--- %9s (%7.4f GB): alias to embedding", "out", 0.);
  } else {
    snprintf(s, sizeof(s), "--- %9s (%7.4f GB)", "out", out_gb);
    util_matrix_summary(s, 1, out_len, 3, w->out_weight);
  }
}

// Allocate a logits buffer for a given number of tokens
float* transformer_logits_malloc(
    transformer_t* transformer, size_t logits_count, size_t* vocabulary_len
) {
  if (!transformer || !vocabulary_len) {
    return NULL;
  }

  *vocabulary_len = transformer->config->vocabulary_len;
  size_t len = logits_count * (*vocabulary_len);
  float* logits = malloc(len * sizeof(*logits));
  if (logits == NULL) {
    UTIL_DIE("failed to malloc for logits");
  }
  return logits;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// RMSNorm (Root Mean Square Normalization) + scaling operation
// y = (x / sqrt(mean(x**2) + epsilon)) * w
static void rmsnorm(
    size_t sequence_len,
    size_t embedding_dim,
    float y[sequence_len][embedding_dim],
    float x[sequence_len][embedding_dim],
    uint16_t w[embedding_dim],
    float epsilon
) {
  for (size_t i = 0; i < sequence_len; i++) {
    // Calculate sum of squares
    float ss = 0.0f;
    for (size_t j = 0; j < embedding_dim; j++) {
      ss += x[i][j] * x[i][j];
    }
    ss /= embedding_dim;
    ss += epsilon;
    ss = 1.0f / sqrtf(ss);
    // Normalize and scale
    for (size_t j = 0; j < embedding_dim; j++) {
      y[i][j] = util_bf16_to_f32(w[j]) * (ss * x[i][j]);
    }
  }
}

// Softmax operation on rows of x:
// converts a vector of real numbers into a probability distribution
// where each value is in the range ]0, 1[ and the sum is 1, such that
// larger input values correspond to larger output probabilities.
static void softmax(
    size_t sequence_len,
    size_t past,
    size_t context_len,
    float x[sequence_len][context_len]
) {
  for (size_t i = 0; i < sequence_len; i++) {
    // Find max value (for numerical stability)
    float max_val = x[i][0];
    for (size_t j = 1; j < past + i + 1; j++) {
      if (x[i][j] > max_val) {
        max_val = x[i][j];
      }
    }
    // Exp and sum
    float sum = 0.0f;
    for (size_t j = 0; j < past + i + 1; j++) {
      x[i][j] = expf(x[i][j] - max_val);
      sum += x[i][j];
    }
    // Normalize
    for (size_t j = 0; j < past + i + 1; j++) {
      x[i][j] /= sum;
    }
  }
}

// Matrix multiplication
void matmul(
    size_t row_count,
    size_t col_count,
    size_t red_count,
    float y[row_count][col_count],
    float x[row_count][red_count],
    uint16_t w[col_count][red_count]
) {
  for (size_t i = 0; i < row_count; i++) {
    for (size_t j = 0; j < col_count; j++) {
      y[i][j] = 0.0f;
      for (size_t k = 0; k < red_count; k++) {
        y[i][j] += x[i][k] * util_bf16_to_f32(w[j][k]);
      }
    }
  }
}

// Apply RoPE (Rotary Positional Embedding) to a set of vectors,
// rotateing them in the complex plane according to their position
void rope(
    size_t context_len,
    size_t sequence_len,
    size_t head_dim,
    float x[sequence_len][head_dim],
    float rope_cos_sin[context_len][head_dim],
    size_t cached_count
) {
  for (size_t i = 0; i < sequence_len; i++) {
    for (size_t j = 0; j < head_dim; j += 2) {
      float fr = rope_cos_sin[cached_count + i][j];
      float fi = rope_cos_sin[cached_count + i][j + 1];
      float v0 = x[i][j];
      float v1 = x[i][j + 1];
      x[i][j]     = v0 * fr - v1 * fi;
      x[i][j + 1] = v0 * fi + v1 * fr;
    }
  }
}

// Here is the compute function. Yep, LLMs are just that simple :)!
// Execute the transformer model on a chunk of tokens, i.e. computes the
// logits (unnormalized probability distribution) for the next token(s)
// given a chunk of input tokens and the cached state from previous tokens
static void transformer_predict_chunk(
    // Input
    size_t token_count,
    int* token,
    // Configuration
    size_t vocabulary_len,
    size_t context_len,
    size_t layer_count,
    size_t q_head_count,
    size_t kv_head_count,
    size_t q_head_per_kv_head_count,
    size_t embedding_dim,
    size_t head_dim,
    size_t hidden_dim,
    float epsilon,
    // Weights
    uint16_t embedding_weight[restrict vocabulary_len][embedding_dim],
    uint16_t mha_norm_weight[restrict layer_count][embedding_dim],
    uint16_t mha_q_weight[restrict layer_count][kv_head_count]
                         [q_head_per_kv_head_count][head_dim][embedding_dim],
    uint16_t mha_k_weight[restrict layer_count][kv_head_count][head_dim]
                         [embedding_dim],
    uint16_t mha_v_weight[restrict layer_count][kv_head_count][head_dim]
                         [embedding_dim],
    uint16_t mha_out_weight[restrict layer_count][embedding_dim][embedding_dim],
    uint16_t ffn_norm_weight[restrict layer_count][embedding_dim],
    uint16_t ffn_fc_weight[restrict layer_count][hidden_dim][embedding_dim],
    uint16_t ffn_up_weight[restrict layer_count][hidden_dim][embedding_dim],
    uint16_t ffn_out_weight[restrict layer_count][embedding_dim][hidden_dim],
    uint16_t out_norm_weight[restrict embedding_dim],
    uint16_t out_weight[restrict vocabulary_len][embedding_dim],
    // State
    float embedding[restrict TRANSFORMER_CHUNK_MAX_LEN][embedding_dim],
    float mha_norm[restrict TRANSFORMER_CHUNK_MAX_LEN][embedding_dim],
    float mha_q[restrict kv_head_count][q_head_per_kv_head_count]
               [TRANSFORMER_CHUNK_MAX_LEN][head_dim],
    float mha_score[restrict kv_head_count][q_head_per_kv_head_count]
                   [TRANSFORMER_CHUNK_MAX_LEN][context_len],
    float mha_att[restrict TRANSFORMER_CHUNK_MAX_LEN][kv_head_count]
                 [q_head_per_kv_head_count][head_dim],
    float mha_out[restrict TRANSFORMER_CHUNK_MAX_LEN][embedding_dim],
    float ffn_norm[restrict TRANSFORMER_CHUNK_MAX_LEN][embedding_dim],
    float ffn_fc[restrict TRANSFORMER_CHUNK_MAX_LEN][hidden_dim],
    float ffn_up[restrict TRANSFORMER_CHUNK_MAX_LEN][hidden_dim],
    float ffn_out[restrict TRANSFORMER_CHUNK_MAX_LEN][embedding_dim],
    size_t cached_count,
    float k_cache[restrict layer_count][kv_head_count][context_len][head_dim],
    float v_cache[restrict layer_count][kv_head_count][context_len][head_dim],
    float rope_cos_sin[restrict context_len][head_dim],
    // Output
    size_t logits_count,
    float logits[restrict TRANSFORMER_CHUNK_MAX_LEN][vocabulary_len]
) {
  (void)q_head_count; // Unused except in debug mode

  // Convert token ids to embedding vector representation
  for (size_t t = 0; t < token_count; t++) {
    for (size_t e = 0; e < embedding_dim; e++) {
      embedding[t][e] = util_bf16_to_f32(embedding_weight[token[t]][e]);
    }
  }

  // Execute decoder layers
  for (size_t l = 0; l < layer_count; l++) {
    // Attention rmsnorm: normalize the embedding vectors for the current layer
    rmsnorm(
        token_count,
        embedding_dim,
        mha_norm,
        embedding,
        mha_norm_weight[l],
        epsilon
    );

    // Q matmul for all Q-heads
    for (size_t k = 0; k < kv_head_count; k++) {
      for (size_t q = 0; q < q_head_per_kv_head_count; q++) {
        for (size_t t = 0; t < token_count; t++) {
          for (size_t h = 0; h < head_dim; h++) {
            mha_q[k][q][t][h] = 0.0f;
            for (size_t e = 0; e < embedding_dim; e++) {
              mha_q[k][q][t][h] +=
                  mha_norm[t][e] *
                  util_bf16_to_f32(mha_q_weight[l][k][q][h][e]);
            }
          }
        }
      }
    }

    // K matmul for all KV-heads, storing in the k_cache
    for (size_t k = 0; k < kv_head_count; k++) {
      for (size_t t = 0; t < token_count; t++) {
        for (size_t h = 0; h < head_dim; h++) {
          k_cache[l][k][cached_count + t][h] = 0.0f;
          for (size_t e = 0; e < embedding_dim; e++) {
            k_cache[l][k][cached_count + t][h] +=
                mha_norm[t][e] * util_bf16_to_f32(mha_k_weight[l][k][h][e]);
          }
        }
      }
    }

    // V matmul for all KV-heads, storing in the v_cache
    for (size_t k = 0; k < kv_head_count; k++) {
      for (size_t t = 0; t < token_count; t++) {
        for (size_t h = 0; h < head_dim; h++) {
          v_cache[l][k][cached_count + t][h] = 0.0f;
          for (size_t e = 0; e < embedding_dim; e++) {
            v_cache[l][k][cached_count + t][h] +=
                mha_norm[t][e] * util_bf16_to_f32(mha_v_weight[l][k][h][e]);
          }
        }
      }
    }

    // RoPE Q for all Q-heads: complex-valued rotate Q in each head
    for (size_t k = 0; k < kv_head_count; k++) {
      for (size_t q = 0; q < q_head_per_kv_head_count; q++) {
        for (size_t t = 0; t < token_count; t++) {
          for (size_t h = 0; h < head_dim; h += 2) {
            float fr = rope_cos_sin[cached_count + t][h + 0];
            float fi = rope_cos_sin[cached_count + t][h + 1];
            float v0 = mha_q[k][q][t][h + 0];
            float v1 = mha_q[k][q][t][h + 1];
            mha_q[k][q][t][h + 0] = v0 * fr - v1 * fi;
            mha_q[k][q][t][h + 1] = v0 * fi + v1 * fr;
          }
        }
      }
    }

    // RoPE K for all KV-heads: complex-valued rotate K in each head
    for (size_t k = 0; k < kv_head_count; k++) {
      for (size_t t = 0; t < token_count; t++) {
        for (size_t h = 0; h < head_dim; h += 2) {
          float fr = rope_cos_sin[cached_count + t][h + 0];
          float fi = rope_cos_sin[cached_count + t][h + 1];
          float v0 = k_cache[l][k][cached_count + t][h + 0];
          float v1 = k_cache[l][k][cached_count + t][h + 1];
          k_cache[l][k][cached_count + t][h + 0] = v0 * fr - v1 * fi;
          k_cache[l][k][cached_count + t][h + 1] = v0 * fi + v1 * fr;
        }
      }
    }

    // Multihead attention. iterate over all Q-heads
    for (size_t k = 0; k < kv_head_count; k++) {
      for (size_t q = 0; q < q_head_per_kv_head_count; q++) {
        for (size_t t = 0; t < token_count; t++) {
          // Calculate the attention score: QKˆT / sqrt(head_dim)
          // Here we don't use mask but a triangular loop (no compute
          // for future tokens)
          for (size_t p = 0; p < cached_count + t + 1; p++) {
            mha_score[k][q][t][p] = 0.0f;
            for (size_t h = 0; h < head_dim; h++) {
              mha_score[k][q][t][p] += mha_q[k][q][t][h] * k_cache[l][k][p][h];
            }
            mha_score[k][q][t][p] /= sqrtf(head_dim);
          }
        }

        // Softmax the scores to get attention weights
        softmax(token_count, cached_count, context_len, mha_score[k][q]);

        for (size_t t = 0; t < token_count; t++) {
          // Weighted sum of the values, here the access function of
          // mha_att is to please the output matmul
          for (size_t h = 0; h < head_dim; h++) {
            mha_att[t][k][q][h] = 0.0f;
          }
          for (size_t p = 0; p <= cached_count + t; p++) {
            for (size_t h = 0; h < head_dim; h++) {
              mha_att[t][k][q][h] +=
                  mha_score[k][q][t][p] * v_cache[l][k][p][h];
            }
          }
        }
      }
    }

    // Final matmul to get the output of the attention
    // Note we reshape mha_att[t][k][q][h] to mha_att[t][kqh] with
    // 0 <= kqh < embedding_dim (just casting because memory layout is ok)
    for (size_t t = 0; t < token_count; t++) {
      for (size_t e = 0; e < embedding_dim; e++) {
        mha_out[t][e] = 0.0f;
        for (size_t kqh = 0; kqh < embedding_dim; kqh++) {
          mha_out[t][e] += ((float (*)[embedding_dim])mha_att)[t][kqh] *
                           util_bf16_to_f32(mha_out_weight[l][e][kqh]);
        }
      }
    }

    // Residual connection back into x
    for (size_t t = 0; t < token_count; t++) {
      for (size_t e = 0; e < embedding_dim; e++) {
        embedding[t][e] += mha_out[t][e];
      }
    }

    // Feed-forward network's rmsnorm
    rmsnorm(
        token_count,
        embedding_dim,
        ffn_norm,
        embedding,
        ffn_norm_weight[l],
        epsilon
    );

    // Feed-forward's fully-connected matmul (a.k.a. gate)
    for (size_t t = 0; t < token_count; t++) {
      for (size_t h = 0; h < hidden_dim; h++) {
        ffn_fc[t][h] = 0.0f;
        for (size_t e = 0; e < embedding_dim; e++) {
          ffn_fc[t][h] +=
              ffn_norm[t][e] * util_bf16_to_f32(ffn_fc_weight[l][h][e]);
        }
      }
    }

    // Feed-forward's up matmul
    for (size_t t = 0; t < token_count; t++) {
      for (size_t h = 0; h < hidden_dim; h++) {
        ffn_up[t][h] = 0.0f;
        for (size_t e = 0; e < embedding_dim; e++) {
          ffn_up[t][h] +=
              ffn_norm[t][e] * util_bf16_to_f32(ffn_up_weight[l][h][e]);
        }
      }
    }

    // SwiGLU non-linearity
    for (size_t t = 0; t < token_count; t++) {
      for (size_t e = 0; e < hidden_dim; e++) {
        // SiLU(x)=x*σ(x), where σ(x) is the logistic sigmoid
        ffn_fc[t][e] *= (1.0f / (1.0f + expf(-ffn_fc[t][e])));
        // Elementwise multiply with ffn_up_weight(x)
        ffn_fc[t][e] *= ffn_up[t][e];
      }
    }

    // Final matmul to get the output of the feed-forward network
    for (size_t t = 0; t < token_count; t++) {
      for (size_t e = 0; e < embedding_dim; e++) {
        ffn_out[t][e] = 0.0f;
        for (size_t h = 0; h < hidden_dim; h++) {
          ffn_out[t][e] +=
              ffn_fc[t][h] * util_bf16_to_f32(ffn_out_weight[l][e][h]);
        }
      }
    }

    // Residual connection
    for (size_t t = 0; t < token_count; t++) {
      for (size_t e = 0; e < embedding_dim; e++) {
        embedding[t][e] += ffn_out[t][e];
      }
    }

    #ifdef DEBUG
    if (l == 0 || l == layer_count - 1) {
      size_t mha_len = token_count * embedding_dim;
      size_t hidden_len = token_count * hidden_dim;
      size_t score_len = q_head_count * token_count * context_len;
      fprintf(stderr, "Transformer state at layer %zu:\n", l);
      util_matrix_summary("- embedding", 1, mha_len, 3, (float*)embedding);
      util_matrix_summary("-  mha_norm", 1, mha_len, 3, (float*)mha_norm);
      util_matrix_summary("-     mha_q", 1, mha_len, 3, (float*)mha_q);
      util_matrix_summary("- mha_score", 1, score_len, 3, (float*)mha_score);
      util_matrix_summary("-   mha_att", 1, mha_len, 3, (float*)mha_att);
      util_matrix_summary("-   mha_out", 1, mha_len, 3, (float*)mha_out);
      util_matrix_summary("-  ffn_norm", 1, mha_len, 3, (float*)ffn_norm);
      util_matrix_summary("-    ffn_fc", 1, hidden_len, 3, (float*)ffn_fc);
      util_matrix_summary("-    ffn_up", 1, hidden_len, 3, (float*)ffn_up);
      util_matrix_summary("-   ffn_out", 1, mha_len, 3, (float*)ffn_out);
    }
    #endif
  }

  // Final rmsnorm
  rmsnorm(
      token_count,
      embedding_dim,
      embedding,
      embedding,
      out_norm_weight,
      epsilon
  );

  // Classifier into logits
  for (size_t l = 0; l < logits_count; l++) {
    for (size_t v = 0; v < vocabulary_len; v++) {
      logits[l][v] = 0.0f;
      for (size_t e = 0; e < embedding_dim; e++) {
        logits[l][v] += embedding[l + token_count - logits_count][e] *
                        util_bf16_to_f32(out_weight[v][e]);
      }
    }
  }
}

// Main function to run the transformer model on a sequence of tokens
// and produce the logits for the next token(s). Handles chunking of input
// tokens to limit the memory usage.
void transformer_predict(
    transformer_t* transformer,
    size_t token_count,
    int* token,
    size_t logits_count,
    float* logits
) {
  if (!transformer || !logits || token_count == 0 || logits_count == 0) {
    return;
  }

  transformer_configuration_t* c = transformer->config;
  transformer_weights_t* w = transformer->weights;
  transformer_state_t* s = transformer->state;

  if (token_count + s->cached_count > c->context_len) {
    UTIL_DIE("context length exhausted");
  }

  size_t vocabulary_len = c->vocabulary_len;
  size_t context_len = c->context_len;
  size_t layer_count = c->layer_count;
  size_t q_head_count = c->q_head_count;
  size_t kv_head_count = c->kv_head_count;
  size_t q_head_per_kv_head_count = q_head_count / kv_head_count;
  size_t embedding_dim = c->embedding_dim;
  size_t head_dim = embedding_dim / q_head_count;
  size_t hidden_dim = c->hidden_dim;

  // Clamp logits_count to available positions
  if (logits_count > token_count) {
    logits_count = token_count;
  }

  size_t logits_start = token_count - logits_count;

  for (size_t t = 0; t < token_count; t += TRANSFORMER_CHUNK_MAX_LEN) {
    // Number of tokens processed by this chunk:
    size_t chunk_token_count =
        UTIL_MIN(TRANSFORMER_CHUNK_MAX_LEN, token_count - t);

    // Base pointer for this chunk input tokens
    int* chunk_token = token + t;

    // Number of logits this chunk needs to compute:
    // chunk covers token positions [t, t + chunk_token_count[
    // compute logits for the intersection with [logits_start, token_count[
    size_t compute_start_pos = UTIL_MAX(t, logits_start);
    size_t compute_end_pos = UTIL_MIN(t + chunk_token_count, token_count);
    size_t chunk_logits_count = compute_end_pos - compute_start_pos;
    if (compute_start_pos >= compute_end_pos) {
      chunk_logits_count = 0;
    }

    // Row in logits where this chunk should write its first logits
    // (in [0, logits_count[):
    size_t chunk_logits_row_offset = compute_start_pos - logits_start;

    // Base pointer for this chunk logits
    float* chunk_logits = logits + chunk_logits_row_offset * vocabulary_len;

    transformer_predict_chunk(
        chunk_token_count,
        chunk_token,

        vocabulary_len,
        context_len,
        layer_count,
        q_head_count,
        kv_head_count,
        q_head_per_kv_head_count,
        embedding_dim,
        head_dim,
        hidden_dim,
        1e-5f,

        (uint16_t (*)[embedding_dim])w->embedding_weight,
        (uint16_t (*)[embedding_dim])w->mha_norm_weight,
        (uint16_t (*)[kv_head_count][q_head_per_kv_head_count][head_dim]
                     [embedding_dim])w->mha_q_weight,
        (uint16_t (*)[kv_head_count][head_dim][embedding_dim])w->mha_k_weight,
        (uint16_t (*)[kv_head_count][head_dim][embedding_dim])w->mha_v_weight,
        (uint16_t (*)[embedding_dim][embedding_dim])w->mha_out_weight,
        (uint16_t (*)[embedding_dim])w->ffn_norm_weight,
        (uint16_t (*)[hidden_dim][embedding_dim])w->ffn_fc_weight,
        (uint16_t (*)[hidden_dim][embedding_dim])w->ffn_up_weight,
        (uint16_t (*)[embedding_dim][hidden_dim])w->ffn_out_weight,
        (uint16_t(*))w->out_norm_weight,
        (uint16_t (*)[embedding_dim])w->out_weight,

        (float (*)[embedding_dim])s->embedding,
        (float (*)[embedding_dim])s->mha_norm,
        (float (*)[q_head_per_kv_head_count][TRANSFORMER_CHUNK_MAX_LEN]
                  [head_dim])s->mha_q,
        (float (*)[q_head_per_kv_head_count][context_len][context_len])
            s->mha_score,
        (float (*)[kv_head_count][q_head_per_kv_head_count][head_dim])
            s->mha_att,
        (float (*)[embedding_dim])s->mha_out,
        (float (*)[embedding_dim])s->ffn_norm,
        (float (*)[hidden_dim])s->ffn_fc,
        (float (*)[hidden_dim])s->ffn_up,
        (float (*)[embedding_dim])s->ffn_out,
        s->cached_count,
        (float (*)[kv_head_count][context_len][head_dim])s->k_cache,
        (float (*)[kv_head_count][context_len][head_dim])s->v_cache,
        (float (*)[head_dim])s->rope_cos_sin,

        chunk_logits_count,
        (float (*)[vocabulary_len])chunk_logits
    );

    s->cached_count += chunk_token_count;
  }
}
