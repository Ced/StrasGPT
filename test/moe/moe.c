// Test-only driver: inspect the final layer and all token logits.
#include "../../source/transformer.c"
#include "options.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static int check_dot_mxfp4(void) {
  // Isolate every code at every position, including extreme scales.
  float activation[32] = {0};
  uint8_t block[1][16] = {{0}};
  for (size_t s = 0; s < 256; s++) {
    uint8_t scale[1] = {s};
    for (size_t code = 0; code < 16; code++) {
      float expected = util_mxfp4_to_f32(code, s);
      for (size_t i = 0; i < 32; i++) {
        activation[i] = 1.0f;
        block[0][i / 2] = code << (4 * (i % 2));
        float actual = dot_mxfp4(32, activation, block, scale);
        if (!(isnan(expected) ? isnan(actual) : actual == expected)) {
          fprintf(stderr, "MXFP4 dot: scale=%zu code=%zu position=%zu "
                          "actual=%g expected=%g\n",
                  s, code, i, actual, expected);
          return 1;
        }
        activation[i] = 0.0f;
        block[0][i / 2] = 0;
      }
    }
  }

  // Mixed signs/scales, cancellation, odd block counts and long rows.
  size_t block_count[] = {1, 2, 3, 7, 90, 128, 513};
  uint32_t random = 42;
  for (size_t n = 0; n < sizeof(block_count) / sizeof(*block_count); n++) {
    size_t len = block_count[n] * 32;
    // Offset allocations to exercise loads without SIMD alignment.
    float* storage = malloc((len + 1) * sizeof(float));
    uint8_t* packed = malloc(len / 2 + 1);
    uint8_t* scales = malloc(block_count[n] + 1);
    if (!storage || !packed || !scales) UTIL_DIE("failed to malloc for test");
    float* a = storage + 1;
    uint8_t (*w)[16] = (uint8_t (*)[16])(packed + 1);
    uint8_t* s = scales + 1;
    for (size_t trial = 0; trial < 32; trial++) {
      double expected = 0.0;
      double magnitude = 0.0;
      for (size_t b = 0; b < block_count[n]; b++) {
        s[b] = 115 + (b + trial) % 24;
        for (size_t i = 0; i < 16; i++) {
          random = random * 1664525u + 1013904223u;
          w[b][i] = random >> 24;
        }
        for (size_t i = 0; i < 32; i++) {
          random = random * 1664525u + 1013904223u;
          a[b * 32 + i] = ((int)(random >> 8) - 8388608) / 8388608.0f;
          uint8_t code = w[b][i / 2] >> (4 * (i % 2));
          double product = (double)a[b * 32 + i] *
                           util_mxfp4_to_f32(code, s[b]);
          expected += product;
          magnitude += fabs(product);
        }
      }
      float actual = dot_mxfp4(len, a, w, s);
      if (!isfinite(actual) ||
          fabs(actual - expected) > 2e-6 * magnitude + 1e-6) {
        fprintf(stderr, "MXFP4 dot: len=%zu trial=%zu "
                        "actual=%g expected=%g\n",
                len, trial, actual, expected);
        free(storage);
        free(packed);
        free(scales);
        return 1;
      }
    }
    free(storage);
    free(packed);
    free(scales);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && strcmp(argv[1], "dot") == 0) {
    return check_dot_mxfp4();
  }
  if (argc == 2 && strcmp(argv[1], "decode") == 0) {
    for (size_t scale = 0; scale < 256; scale++) {
      for (size_t code = 0; code < 16; code++) {
        float value = util_mxfp4_to_f32(code, scale);
        if (fwrite(&value, sizeof(value), 1, stdout) != 1) return 1;
      }
    }
    return 0;
  }
  if (argc != 6) {
    fprintf(stderr, "usage: %s MODEL COUNT CHUNK LAYERS THREADS\n", argv[0]);
    return 1;
  }
  options_t options = {0};
  options.model_dir = argv[1];
  safetensors_t* tensors = safetensors_read(&options);
  transformer_t* model = transformer_from_safetensors(tensors);
  transformer_configuration_t* c = model->config;
  transformer_state_t* s = model->state;
  if (atoi(argv[4])) c->layer_count = atoi(argv[4]);
  size_t count = atoi(argv[2]);
  size_t chunk = atoi(argv[3]);
  if (!chunk) chunk = count;
  int* tokens = malloc(count * sizeof(int));
  float* logits = malloc(count * c->vocabulary_len * sizeof(float));
  if (!tokens || !logits) UTIL_DIE("failed to malloc for test");
  for (size_t i = 0; i < count; i++) tokens[i] = (i * 7 + 1) % 32;
#ifdef _OPENMP
  omp_set_num_threads(atoi(argv[5]));
#endif
  size_t last = 0;
  for (size_t i = 0; i < count; i += chunk) {
    size_t len = UTIL_MIN(chunk, count - i);
#pragma omp parallel
    transformer_predict(model, len, tokens + i, len,
                        logits + i * c->vocabulary_len);
    last = (len - 1) % TRANSFORMER_CHUNK_MAX_LEN;
  }
  size_t selected = c->expert_per_token_count;
  size_t hidden_len = selected * c->hidden_dim;
  size_t out_len = selected * c->embedding_dim;
  fwrite(s->ffn_norm + last * c->embedding_dim,
         sizeof(float), c->embedding_dim, stdout);
  fwrite(s->ffn_router_logits + last * c->expert_count,
         sizeof(float), c->expert_count, stdout);
  fwrite(s->ffn_router_index + last * selected,
         sizeof(size_t), selected, stdout);
  fwrite(s->ffn_router_score + last * selected,
         sizeof(float), selected, stdout);
  fwrite(s->ffn_xp_gate + last * hidden_len, sizeof(float), hidden_len, stdout);
  fwrite(s->ffn_xp_up + last * hidden_len, sizeof(float), hidden_len, stdout);
  fwrite(s->ffn_xp_fc + last * hidden_len, sizeof(float), hidden_len, stdout);
  fwrite(s->ffn_xp_out + last * out_len, sizeof(float), out_len, stdout);
  fwrite(s->ffn_out + last * c->embedding_dim,
         sizeof(float), c->embedding_dim, stdout);
  fwrite(logits, sizeof(float), count * c->vocabulary_len, stdout);
  free(tokens);
  free(logits);
  transformer_free(model);
  safetensors_free(tensors);
  return ferror(stdout) ? 1 : 0;
}
