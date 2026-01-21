#ifndef SAFETENSORS_H
# define SAFETENSORS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct options;

#define SAFETENSORS_FILE_CONFIG      "config.json"
#define SAFETENSORS_FILE_INDEX       "model.safetensors.index.json"
#define SAFETENSORS_FILE_SAFETENSORS "model.safetensors"

#define SAFETENSORS_PATTERN_EMBEDDING_WEIGHT  "model.embed_tokens.weight"
#define SAFETENSORS_PATTERN_MHA_NORM_WEIGHT   "model.layers.%d.input_layernorm.weight"
#define SAFETENSORS_PATTERN_MHA_Q_WEIGHT      "model.layers.%d.self_attn.q_proj.weight"
#define SAFETENSORS_PATTERN_MHA_Q_NORM_WEIGHT "model.layers.%d.self_attn.q_norm.weight"
#define SAFETENSORS_PATTERN_MHA_K_WEIGHT      "model.layers.%d.self_attn.k_proj.weight"
#define SAFETENSORS_PATTERN_MHA_K_NORM_WEIGHT "model.layers.%d.self_attn.k_norm.weight"
#define SAFETENSORS_PATTERN_MHA_V_WEIGHT      "model.layers.%d.self_attn.v_proj.weight"
#define SAFETENSORS_PATTERN_MHA_OUT_WEIGHT    "model.layers.%d.self_attn.o_proj.weight"
#define SAFETENSORS_PATTERN_FFN_NORM_WEIGHT   "model.layers.%d.post_attention_layernorm.weight"
#define SAFETENSORS_PATTERN_FFN_FC_WEIGHT     "model.layers.%d.mlp.gate_proj.weight"
#define SAFETENSORS_PATTERN_FFN_UP_WEIGHT     "model.layers.%d.mlp.up_proj.weight"
#define SAFETENSORS_PATTERN_FFN_OUT_WEIGHT    "model.layers.%d.mlp.down_proj.weight"
#define SAFETENSORS_PATTERN_OUT_NORM_WEIGHT   "model.norm.weight"
#define SAFETENSORS_PATTERN_OUT_WEIGHT        "lm_head.weight"

#define SAFETENSORS_MAX_FILE_COUNT          64
#define SAFETENSORS_MAX_DIM_COUNT           8
#define SAFETENSORS_MAX_MROPE_SECTION_COUNT 8
#define SAFETENSORS_MAX_TENSOR_COUNT        65536
#define SAFETENSORS_MAX_STRING              1024

typedef enum {
  SAFETENSORS_TYPE_F16,  // IEEE float16 (half precision)
  SAFETENSORS_TYPE_BF16, // bfloat16 (truncated mantissa)
  SAFETENSORS_TYPE_F32   // float
} safetensors_type_t;

extern const char *safetensors_type_str[];

typedef struct safetensors_tensor_t {
  char* name;              // Tensor name
  safetensors_type_t type; // Data type
  size_t dim_count;        // Number of dimensions
  size_t dim[SAFETENSORS_MAX_DIM_COUNT]; // Dimensions
  size_t size;             // Size in bytes
  size_t file;             // Index of the file where the tensor is stored
  size_t offset;           // Offset in the file where the tensor data starts
} safetensors_tensor_t;

typedef struct safetensors{
  // Model configuration
  char* model_type;      // Model type string
  size_t embedding_dim;  // Token representation (embedding) dimension
  size_t head_dim;       // Dimensionality of each individual attention head
  size_t hidden_dim;     // Intermediate representation dimension in the FFN
  size_t layer_count;    // Number of decoder layers
  size_t q_head_count;   // Number of query heads
  size_t kv_head_count;  // Number of key/value heads
  size_t vocabulary_len; // Vocabulary size
  size_t context_len;    // Maximum sequence length
  float epsilon;         // RMSNorm epsilon value
  float rope_theta;      // RoPE base frequency
  bool rope_interleaved; // true: pairs interleaved (Meta), false: grouped (HF)
  size_t mrope_section_count; // Sections for multi-scale RoPE (0 if none)
  size_t mrope_section[SAFETENSORS_MAX_MROPE_SECTION_COUNT];

  // Special tokens from the configuration file
  int bos_token_id;      // Beginning of string token id
  int eos_token_id;      // End of string token id

  // File names where tensors are stored
  size_t file_count;
  char* file[SAFETENSORS_MAX_FILE_COUNT];

  // Tensors
  size_t tensor_count;
  safetensors_tensor_t tensor[SAFETENSORS_MAX_TENSOR_COUNT];
} safetensors_t;

safetensors_t* safetensors_malloc(void);
void safetensors_free(safetensors_t* safetensors);
void safetensors_print(FILE* f, const safetensors_t* safetensors);
void safetensors_print_model_infos(FILE* f, const safetensors_t* s);
safetensors_t* safetensors_read(struct options* options);
safetensors_type_t safetensors_type_from_string(const char *s);
void safetensors_file_lookup(safetensors_t* s, char* path, char* file);
size_t safetensors_sizeof(safetensors_type_t type);
bool safetensors_aliased_out_weight(const safetensors_t* safetensors);

#endif // SAFETENSORS_H
