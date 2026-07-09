#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

struct safetensors;

#define TRANSFORMER_CHUNK_MAX_LEN 512

typedef enum {
  TRANSFORMER_LAYER_TYPE_FA, // Full attention
  TRANSFORMER_LAYER_TYPE_LA, // Linear attention
} transformer_layer_type_t;

typedef struct transformer_configuration {
  size_t embedding_dim;    // Token representation (embedding) dimension
  size_t head_dim;         // Dimensionality of each individual attention head
  size_t hidden_dim;       // Intermediate representation dimension in the FFN
  size_t layer_count;      // Number of decoder layers
  size_t q_head_count;     // Number of query heads
  size_t kv_head_count;    // Number of key/value heads
  size_t vocabulary_len;   // Vocabulary size
  size_t context_len;      // Maximum sequence length
  float epsilon;           // RMSNorm epsilon value
  float rope_theta;        // RoPE base frequency
  size_t rope_pair_bound;  // Interleaved: head_dim, half-split: head_dim / 2
  size_t rope_pair_offset; // Interleaved: 1, half-split: head_dim / 2
  size_t rope_pair_stride; // Interleaved: 2, half-split: 1
  size_t mrope_section_count; // Number of multi-scale RoPE section (0 if none)
  size_t* mrope_section;      // Sections for multi-scale RoPE (NULL if none)
  size_t n_rot;               // Number of rotated dimensions per head (RoPE)
  transformer_layer_type_t* layer_type; // [layer_count]
  size_t* layer_compact_index;          // [layer_count]
  size_t fa_layer_count;                // Number of full-attention layers
  size_t la_layer_count;                // Number of linear-attention layers
  size_t la_kernel_size;   // Linear-attention convolution kernel size
  size_t la_k_head_dim;    // Linear-attention key head dimension
  size_t la_k_head_count;  // Linear-attention key head count
  size_t la_v_head_dim;    // Linear-attention value head dimension
  size_t la_v_head_count;  // Linear-attention value head count
  bool mha_output_gate;    // Full attention has a learned output gate
  bool aliased_out_weight; // True if out_weight is aliased to embedding_weight
} transformer_configuration_t;

typedef struct transformer_weights {
  // Embedding parameter set
  uint16_t* embedding_weight; // [vocabulary_len][embedding_dim]
  // Decoder parameter set
  // - Multi-head attention
  uint16_t* mha_norm_weight;   // [layer_count][embedding_dim]
  uint16_t* mha_q_weight;      // [fa_layer_count][q_head_count][
                               //  head_dim][embedding_dim]
  uint16_t* mha_gate_weight;   // Optional, same shape as mha_q_weight
  uint16_t* mha_q_norm_weight; // [fa_layer_count][head_dim]
  uint16_t* mha_k_weight;      // [fa_layer_count][kv_head_count][
                               //  head_dim][embedding_dim]
  uint16_t* mha_k_norm_weight; // [fa_layer_count][head_dim]
  uint16_t* mha_v_weight;      // [fa_layer_count][kv_head_count][
                               //  head_dim][embedding_dim]
  uint16_t* mha_out_weight;    // [fa_layer_count][embedding_dim][
                               //  q_head_count * head_dim]
  // - Linear attention
  uint16_t* la_qkv_weight;   // [la_layer_count][la_qkv_dim][embedding_dim]
  uint16_t* la_gate_weight;  // [la_layer_count][la_v_dim][embedding_dim]
  uint16_t* la_alpha_weight; // [la_layer_count][la_v_head_count][embedding_dim]
  uint16_t* la_beta_weight;  // [la_layer_count][la_v_head_count][embedding_dim]
  uint16_t* la_dt_bias;      // [la_layer_count][la_v_head_count]
  float* la_decay_weight;    // [la_layer_count][la_v_head_count], -exp(A_log)
  uint16_t* la_conv_weight;  // [la_layer_count][la_qkv_dim][la_kernel_size]
  float* la_norm_weight;     // [la_layer_count][la_v_head_dim]
  uint16_t* la_out_weight;   // [la_layer_count][embedding_dim][la_v_dim]
  // - Feed-forward network
  uint16_t* ffn_norm_weight; // [layer_count][embedding_dim]
  uint16_t* ffn_fc_weight;   // [layer_count][embedding_dim][hidden_dim]
  uint16_t* ffn_up_weight;   // [layer_count][embedding_dim][hidden_dim]
  uint16_t* ffn_out_weight;  // [layer_count][hidden_dim][embedding_dim]
  // Output parameter set
  uint16_t* out_norm_weight; // [embedding_dim]
  uint16_t* out_weight;      // [vocabulary_len][embedding_dim]
} transformer_weights_t;

typedef struct transformer_state {
  // Activations
  float* embedding; // [chunk_len][embedding_dim]
  float* mha_norm;  // [chunk_len][embedding_dim]
  float* mha_q;     // [kv_head_count][q_head_per_kv_head_count][
                    //  chunk_len][head_dim]
  float* mha_score; // [kv_head_count][q_head_per_kv_head_count][
                    //  chunk_len][context_len]
  float* mha_att;   // [chunk_len][kv_head_count][q_head_per_kv_head_count][
                    //  head_dim]
                    // also sees as [chunk_len][q_head_count * head_dim]
  float* mha_out;   // [chunk_len][embedding_dim]
  float* ffn_norm;  // [chunk_len][embedding_dim]
  float* ffn_fc;    // [chunk_len][hidden_dim]
  float* ffn_up;    // [chunk_len][hidden_dim]
  float* ffn_out;   // [chunk_len][embedding_dim]
  float* logits;    // [chunk_len][vocabulary_len]
  // KV-cache
  size_t cached_count; // Number of tokens currently cached
  float* k_cache;      // [layer_count][kv_head_count][context_len][head_dim]
  float* v_cache;      // [layer_count][kv_head_count][context_len][head_dim]
  // Utility variables
  float* rope_cos_sin; // [context_len][head_dim]
  // linear attention buffers
  float* la_conv_state; // [la_layer_count][la_qkv_dim][la_kernel_size-1]
  float* la_ssm_state;  // [la_layer_count][la_v_head_count][la_v_head_dim][
                        //  la_k_head_dim]
  // optional buffers for linar attention
  float* la_qkv_mixed; // [la_qkv_dim]
  float* la_conv_out;  // [la_qkv_dim]
  float* la_z;         // [la_v_dim]
  float* la_out_flat;  // [la_v_dim]

} transformer_state_t;

typedef struct transformer {
  transformer_configuration_t* config; // Hyperparameters
  transformer_weights_t* weights;      // Weights
  transformer_state_t* state;          // Activations & dynamic state
} transformer_t;

transformer_t* transformer_from_safetensors(struct safetensors* safetensors);
void transformer_free(transformer_t* transformer);
void transformer_print(FILE* f, const transformer_t* safetensors);
float* transformer_logits_malloc(
    transformer_t* transformer, size_t logits_count, size_t* vocabulary_len
);
void transformer_predict(
    transformer_t* transformer,
    size_t token_count,
    int* token,
    size_t logits_count,
    float* logits
);

#endif // TRANSFORMER_H
