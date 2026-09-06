// Test-only driver: inspect the final layer and all token logits.
#include "../../source/transformer.c"
#include "options.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
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
