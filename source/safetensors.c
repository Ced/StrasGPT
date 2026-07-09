#include "safetensors.h"
#include "options.h"
#include "util.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* safetensors_type_str[] = {"F16", "BF16", "F32"};

// Allocate a safetensors_t structure
safetensors_t* safetensors_malloc(void) {
  safetensors_t* safetensors = calloc(1, sizeof(safetensors_t));
  if (!safetensors) {
    UTIL_DIE("failed to malloc for safetensors_t");
  }
  for (size_t i = 0; i < SAFETENSORS_MAX_LAYER_COUNT; i++) {
    safetensors->layer_type[i] = SAFETENSORS_LAYER_TYPE_FA;
  }
  return safetensors;
}

static const char* safetensors_layer_type_string(
    safetensors_layer_type_t layer_type
) {
  return layer_type == SAFETENSORS_LAYER_TYPE_LA ? "LA"  // "linear_attention"
                                                 : "FA"; // "full_attention"
}

static void safetensors_print_layer_types(
    FILE* f, const safetensors_t* safetensors
) {
  fprintf(f, "[");
  for (size_t i = 0; i < safetensors->layer_count; i++) {
    fprintf(f, "%s", safetensors_layer_type_string(safetensors->layer_type[i]));
    fprintf(f, i + 1 == safetensors->layer_count ? "]\n" : ", ");
  }
  if (safetensors->layer_count == 0) {
    fprintf(f, "]\n");
  }
}

// Free a safetensors_t structure
void safetensors_free(safetensors_t* safetensors) {
  if (safetensors) {
    free(safetensors->model_type);
    for (size_t i = 0; i < safetensors->file_count; i++) {
      free(safetensors->file[i]);
    }
    for (size_t i = 0; i < safetensors->tensor_count; i++) {
      free(safetensors->tensor[i].name);
    }
    free(safetensors);
  }
}

// Print a safetensors_t structure
void safetensors_print(FILE* f, const safetensors_t* safetensors) {
  if (!safetensors) {
    fprintf(f, "safetensors: NULL\n");
    return;
  }

  // Print files
  fprintf(f, "Safetensors:\n");
  fprintf(f, "- Configuration:\n");
  if (safetensors->model_type) {
    fprintf(f, "--- model_type:       %s\n", safetensors->model_type);
  } else {
    fprintf(f, "--- model_type:       (null)\n");
  }
  fprintf(f, "--- embedding_dim:    %zu\n", safetensors->embedding_dim);
  fprintf(f, "--- head_dim:         %zu\n", safetensors->head_dim);
  fprintf(f, "--- hidden_dim:       %zu\n", safetensors->hidden_dim);
  fprintf(f, "--- layer_count:      %zu\n", safetensors->layer_count);
  fprintf(f, "--- layer_types:      ");
  safetensors_print_layer_types(f, safetensors);
  fprintf(f, "--- q_head_count:     %zu\n", safetensors->q_head_count);
  fprintf(f, "--- kv_head_count:    %zu\n", safetensors->kv_head_count);
  fprintf(
      f,
      "--- mha_output_gate:  %s\n",
      safetensors->mha_output_gate ? "true" : "false"
  );
  fprintf(f, "--- la_kernel_size:   %zu\n", safetensors->la_kernel_size);
  fprintf(f, "--- la_k_head_dim:    %zu\n", safetensors->la_k_head_dim);
  fprintf(f, "--- la_k_head_count:  %zu\n", safetensors->la_k_head_count);
  fprintf(f, "--- la_v_head_dim:    %zu\n", safetensors->la_v_head_dim);
  fprintf(f, "--- la_v_head_count:  %zu\n", safetensors->la_v_head_count);
  fprintf(f, "--- vocabulary_len:   %zu\n", safetensors->vocabulary_len);
  fprintf(f, "--- context_len:      %zu\n", safetensors->context_len);
  fprintf(f, "--- rope_theta:       %.1f\n", safetensors->rope_theta);
  fprintf(
      f,
      "--- rope_interleaved: %s\n",
      safetensors->rope_interleaved ? "true" : "false"
  );
  fprintf(f, "--- mrope_sections: ");
  if (safetensors->mrope_section_count == 0) {
    fprintf(f, "none\n");
  } else {
    fprintf(f, "[");
    for (size_t i = 0; i < safetensors->mrope_section_count; i++) {
      fprintf(f, "%zu", safetensors->mrope_section[i]);
      if (i == safetensors->mrope_section_count - 1) {
        fprintf(f, "]\n");
      } else {
        fprintf(f, ", ");
      }
    }
  }
  fprintf(f, "--- bos_token_id:   %d\n", safetensors->bos_token_id);
  fprintf(f, "--- eos_token_id:   %d\n", safetensors->bos_token_id);

  fprintf(f, "- Files (%zu):\n", safetensors->file_count);
  for (size_t i = 0; i < safetensors->file_count; i++) {
    fprintf(f, "--- File[%-zu]: %s\n", i, safetensors->file[i]);
  }

  fprintf(f, "- Tensors (%zu):\n", safetensors->tensor_count);
  // Let's prepare to align the output
  size_t max_id_len = 0;
  size_t max_name_len = 0;
  size_t max_type_len = 0;
  size_t max_dim_len = 0;
  size_t max_size_len = 0;
  size_t max_file_len = 0;
  size_t max_offset_len = 0;
  for (size_t i = 0; i < safetensors->tensor_count; i++) {
    const safetensors_tensor_t* tensor = &safetensors->tensor[i];

    size_t id_len = snprintf(NULL, 0, "%zu", i);
    max_id_len = (id_len > max_id_len) ? id_len : max_id_len;

    size_t name_len = strlen(tensor->name);
    max_name_len = (name_len > max_name_len) ? name_len : max_name_len;

    size_t type_len = strlen(safetensors_type_str[tensor->type]);
    max_type_len = (type_len > max_type_len) ? type_len : max_type_len;

    size_t dim_len = 0;
    for (size_t j = 0; j < tensor->dim_count; j++) {
      bool coma = (j != tensor->dim_count - 1);
      dim_len += snprintf(NULL, 0, "%zu%s", tensor->dim[j], coma ? ", " : "");
    }
    max_dim_len = (dim_len > max_dim_len) ? dim_len : max_dim_len;

    size_t size_len = snprintf(NULL, 0, "%zu", tensor->size);
    max_size_len = (size_len > max_size_len) ? size_len : max_size_len;

    size_t file_len = snprintf(NULL, 0, "%zu", tensor->file);
    max_file_len = (file_len > max_file_len) ? file_len : max_file_len;

    size_t offset_len = snprintf(NULL, 0, "%zu", tensor->offset);
    max_offset_len =
        (offset_len > max_offset_len) ? offset_len : max_offset_len;
  }

  for (size_t i = 0; i < safetensors->tensor_count; i++) {
    const safetensors_tensor_t* tensor = &safetensors->tensor[i];
    fprintf(
        f,
        "--- Tensor[%-*zu]: name=%-*s type=%-*s dim=[",
        (int)max_id_len,
        i,
        (int)max_name_len,
        tensor->name,
        (int)max_type_len,
        safetensors_type_str[tensor->type]
    );

    // Format dimensions
    char buf[128];
    int pos = 0;
    for (size_t j = 0; j < tensor->dim_count; j++) {
      bool coma = (j != tensor->dim_count - 1);
      pos += snprintf(
          buf + pos,
          sizeof(buf) - pos,
          "%zu%s",
          tensor->dim[j],
          coma ? ", " : ""
      );
    }

    fprintf(
        f,
        "%-*s] size=%-*zu file=%-*zu offset=%-*zu\n",
        (int)max_dim_len,
        buf,
        (int)max_size_len,
        tensor->size,
        (int)max_file_len,
        tensor->file,
        (int)max_offset_len,
        tensor->offset
    );
  }
}

// Print a safetensors_t structure
void safetensors_print_model_infos(FILE* f, const safetensors_t* s) {
  if (!s) {
    fprintf(f, "model: NULL\n");
    return;
  }

  size_t q_head_per_kv_head_count = s->q_head_count / s->kv_head_count;
  size_t qkv_weight_dim = s->head_dim * s->embedding_dim;
  bool aliased_out_weight = safetensors_aliased_out_weight(s);
  bool mha_output_gate = s->mha_output_gate;
  size_t fa_layer_count = 0;
  size_t la_layer_count = 0;
  for (size_t i = 0; i < s->layer_count; i++) {
    if (s->layer_type[i] == SAFETENSORS_LAYER_TYPE_LA) {
      la_layer_count++;
    } else {
      fa_layer_count++;
    }
  }

  // Print files
  fprintf(f, "Model:\n");
  fprintf(f, "- Configuration:\n");
  fprintf(f, "--- embedding_dim:            %zu\n", s->embedding_dim);
  fprintf(f, "--- head_dim:                 %zu\n", s->head_dim);
  fprintf(f, "--- hidden_dim:               %zu\n", s->hidden_dim);
  fprintf(f, "--- layer_count:              %zu\n", s->layer_count);
  fprintf(f, "--- layer_types:              ");
  safetensors_print_layer_types(f, s);
  fprintf(f, "--- fa_layer_count:           %zu\n", fa_layer_count);
  fprintf(f, "--- la_layer_count:           %zu\n", la_layer_count);
  fprintf(f, "--- q_head_count:             %zu\n", s->q_head_count);
  fprintf(f, "--- kv_head_count:            %zu\n", s->kv_head_count);
  fprintf(
      f,
      "--- mha_output_gate:          %s\n",
      mha_output_gate ? "true" : "false"
  );
  fprintf(f, "--- la_kernel_size:           %zu\n", s->la_kernel_size);
  fprintf(f, "--- la_k_head_dim:            %zu\n", s->la_k_head_dim);
  fprintf(f, "--- la_k_head_count:          %zu\n", s->la_k_head_count);
  fprintf(f, "--- la_v_head_dim:            %zu\n", s->la_v_head_dim);
  fprintf(f, "--- la_v_head_count:          %zu\n", s->la_v_head_count);
  fprintf(f, "--- q_head_per_kv_head_count: %zu\n", q_head_per_kv_head_count);
  fprintf(f, "--- vocabulary_len:           %zu\n", s->vocabulary_len);
  fprintf(f, "--- context_len:              %zu\n", s->context_len);
  fprintf(f, "--- epsilon:                  %.g\n", s->epsilon);
  fprintf(f, "--- rope_theta:               %.1f\n", s->rope_theta);
  fprintf(f, "--- mrope_sections:           ");
  if (s->mrope_section_count == 0) {
    fprintf(f, "none\n");
  } else {
    fprintf(f, "[");
    for (size_t i = 0; i < s->mrope_section_count; i++) {
      fprintf(f, "%zu", s->mrope_section[i]);
      if (i == s->mrope_section_count - 1) {
        fprintf(f, "]\n");
      } else {
        fprintf(f, ", ");
      }
    }
  }

  size_t embedding_len = s->vocabulary_len * s->embedding_dim;
  size_t mha_norm_len = s->layer_count * s->embedding_dim;
  size_t mha_q_len = fa_layer_count * s->q_head_count * qkv_weight_dim;
  size_t mha_kv_len = fa_layer_count * s->kv_head_count * qkv_weight_dim;
  size_t mha_out_dim = s->q_head_count * s->head_dim;
  size_t mha_out_len = fa_layer_count * s->embedding_dim * mha_out_dim;
  size_t la_qkv_dim = 2 * s->la_k_head_count * s->la_k_head_dim +
                      s->la_v_head_count * s->la_v_head_dim;
  size_t la_v_dim = s->la_v_head_count * s->la_v_head_dim;
  size_t la_qkv_len = la_layer_count * la_qkv_dim * s->embedding_dim;
  size_t la_gate_len = la_layer_count * la_v_dim * s->embedding_dim;
  size_t la_alpha_len = la_layer_count * s->la_v_head_count * s->embedding_dim;
  size_t la_beta_len = la_layer_count * s->la_v_head_count * s->embedding_dim;
  size_t la_dt_len = la_layer_count * s->la_v_head_count;
  size_t la_decay_len = la_layer_count * s->la_v_head_count;
  size_t la_conv_len = la_layer_count * la_qkv_dim * s->la_kernel_size;
  size_t la_norm_len = la_layer_count * s->la_v_head_dim;
  size_t la_out_len = la_layer_count * s->embedding_dim * la_v_dim;
  size_t ffn_norm_len = s->layer_count * s->embedding_dim;
  size_t ffn_fc_len = s->layer_count * s->embedding_dim * s->hidden_dim;
  size_t ffn_up_len = s->layer_count * s->embedding_dim * s->hidden_dim;
  size_t ffn_out_len = s->layer_count * s->hidden_dim * s->embedding_dim;
  size_t out_norm_len = s->embedding_dim;
  size_t out_len = s->vocabulary_len * s->embedding_dim;

  double gb = 1024 * 1024 * 1024;
  double embedding_gb = (embedding_len * sizeof(uint16_t)) / gb;
  double mha_norm_gb = (mha_norm_len * sizeof(uint16_t)) / gb;
  double mha_q_gb = (mha_q_len * sizeof(uint16_t)) / gb;
  double mha_gate_gb = mha_output_gate ? mha_q_gb : 0;
  double mha_k_gb = (mha_kv_len * sizeof(uint16_t)) / gb;
  double mha_v_gb = (mha_kv_len * sizeof(uint16_t)) / gb;
  double mha_out_gb = (mha_out_len * sizeof(uint16_t)) / gb;
  double la_qkv_gb = (la_qkv_len * sizeof(uint16_t)) / gb;
  double la_gate_gb = (la_gate_len * sizeof(uint16_t)) / gb;
  double la_alpha_gb = (la_alpha_len * sizeof(uint16_t)) / gb;
  double la_beta_gb = (la_beta_len * sizeof(uint16_t)) / gb;
  double la_dt_gb = (la_dt_len * sizeof(uint16_t)) / gb;
  double la_decay_gb = (la_decay_len * sizeof(float)) / gb;
  double la_conv_gb = (la_conv_len * sizeof(uint16_t)) / gb;
  double la_norm_gb = (la_norm_len * sizeof(float)) / gb;
  double la_out_gb = (la_out_len * sizeof(uint16_t)) / gb;
  double ffn_norm_gb = (ffn_norm_len * sizeof(uint16_t)) / gb;
  double ffn_fc_gb = (ffn_fc_len * sizeof(uint16_t)) / gb;
  double ffn_up_gb = (ffn_up_len * sizeof(uint16_t)) / gb;
  double ffn_out_gb = (ffn_out_len * sizeof(uint16_t)) / gb;
  double out_norm_gb = (out_norm_len * sizeof(uint16_t)) / gb;
  double out_gb = aliased_out_weight ? 0 : (out_len * sizeof(uint16_t)) / gb;

  double total_gb = embedding_gb + mha_norm_gb + mha_q_gb + mha_gate_gb +
                    mha_k_gb + mha_v_gb + mha_out_gb + la_qkv_gb + la_gate_gb +
                    la_alpha_gb + la_beta_gb + la_dt_gb + la_decay_gb +
                    la_conv_gb + la_norm_gb + la_out_gb + ffn_norm_gb +
                    ffn_fc_gb + ffn_up_gb + ffn_out_gb + out_norm_gb + out_gb;

  double non_layer_gb = embedding_gb + out_norm_gb + out_gb;
  double per_layer_gb = (total_gb - non_layer_gb) / s->layer_count;

  fprintf(
      f,
      "- Tensors (total %.2f GB, non-layer %.2f GB, per-layer %.2f GB):\n",
      total_gb,
      non_layer_gb,
      per_layer_gb
  );
  fprintf(
      f,
      "--- embedding (%7.4f GB) [vocabulary_len=%zu][embedding_dim=%zu]\n",
      embedding_gb,
      s->vocabulary_len,
      s->embedding_dim
  );
  fprintf(
      f,
      "---  mha_norm (%7.4f GB) [layer_count=%zu][embedding_dim=%zu]\n",
      mha_norm_gb,
      s->layer_count,
      s->embedding_dim
  );
  if (fa_layer_count > 0) {
    fprintf(
        f,
        "---     mha_q (%7.4f GB) [fa_layer_count=%zu][kv_head_count=%zu]"
        "[q_head_per_kv_head_count=%zu][head_dim=%zu][embedding_dim=%zu]\n",
        mha_q_gb,
        fa_layer_count,
        s->kv_head_count,
        q_head_per_kv_head_count,
        s->head_dim,
        s->embedding_dim
    );
    if (mha_output_gate) {
      fprintf(
          f,
          "---  mha_gate (%7.4f GB) [fa_layer_count=%zu][kv_head_count=%zu]"
          "[q_head_per_kv_head_count=%zu][head_dim=%zu][embedding_dim=%zu]\n",
          mha_gate_gb,
          fa_layer_count,
          s->kv_head_count,
          q_head_per_kv_head_count,
          s->head_dim,
          s->embedding_dim
      );
    }
    fprintf(
        f,
        "---     mha_k (%7.4f GB) [fa_layer_count=%zu][kv_head_count=%zu]"
        "[head_dim=%zu][embedding_dim=%zu]\n",
        mha_k_gb,
        fa_layer_count,
        s->kv_head_count,
        s->head_dim,
        s->embedding_dim
    );
    fprintf(
        f,
        "---     mha_v (%7.4f GB) [fa_layer_count=%zu][kv_head_count=%zu]"
        "[head_dim=%zu][embedding_dim=%zu]\n",
        mha_v_gb,
        fa_layer_count,
        s->kv_head_count,
        s->head_dim,
        s->embedding_dim
    );
    fprintf(
        f,
        "---   mha_out (%7.4f GB) [fa_layer_count=%zu][embedding_dim=%zu]"
        "[q_head_count*head_dim=%zu]\n",
        mha_out_gb,
        fa_layer_count,
        s->embedding_dim,
        s->q_head_count * s->head_dim
    );
  }
  if (la_layer_count > 0) {
    fprintf(
        f,
        "---    la_qkv (%7.4f GB) [la_layer_count=%zu][la_qkv_dim=%zu]"
        "[embedding_dim=%zu]\n",
        la_qkv_gb,
        la_layer_count,
        la_qkv_dim,
        s->embedding_dim
    );
    fprintf(
        f,
        "---   la_gate (%7.4f GB) [la_layer_count=%zu][la_v_dim=%zu]"
        "[embedding_dim=%zu]\n",
        la_gate_gb,
        la_layer_count,
        la_v_dim,
        s->embedding_dim
    );
    fprintf(
        f,
        "---  la_alpha (%7.4f GB) [la_layer_count=%zu][la_v_head_count=%zu]"
        "[embedding_dim=%zu]\n",
        la_alpha_gb,
        la_layer_count,
        s->la_v_head_count,
        s->embedding_dim
    );
    fprintf(
        f,
        "---   la_beta (%7.4f GB) [la_layer_count=%zu][la_v_head_count=%zu]"
        "[embedding_dim=%zu]\n",
        la_beta_gb,
        la_layer_count,
        s->la_v_head_count,
        s->embedding_dim
    );
    fprintf(
        f,
        "---     la_dt (%7.4f GB) [la_layer_count=%zu]"
        "[la_v_head_count=%zu]\n",
        la_dt_gb,
        la_layer_count,
        s->la_v_head_count
    );
    fprintf(
        f,
        "---  la_decay (%7.4f GB) [la_layer_count=%zu]"
        "[la_v_head_count=%zu] (FP32, -exp(A_log))\n",
        la_decay_gb,
        la_layer_count,
        s->la_v_head_count
    );
    fprintf(
        f,
        "---   la_conv (%7.4f GB) [la_layer_count=%zu][la_qkv_dim=%zu]"
        "[la_kernel_size=%zu]\n",
        la_conv_gb,
        la_layer_count,
        la_qkv_dim,
        s->la_kernel_size
    );
    fprintf(
        f,
        "---   la_norm (%7.4f GB) [la_layer_count=%zu]"
        "[la_v_head_dim=%zu] (FP32)\n",
        la_norm_gb,
        la_layer_count,
        s->la_v_head_dim
    );
    fprintf(
        f,
        "---    la_out (%7.4f GB) [la_layer_count=%zu][embedding_dim=%zu]"
        "[la_v_dim=%zu]\n",
        la_out_gb,
        la_layer_count,
        s->embedding_dim,
        la_v_dim
    );
  }
  fprintf(
      f,
      "---  ffn_norm (%7.4f GB) [layer_count=%zu][embedding_dim=%zu]\n",
      ffn_norm_gb,
      s->layer_count,
      s->embedding_dim
  );
  fprintf(
      f,
      "---    ffn_fc (%7.4f GB) [layer_count=%zu][hidden_dim=%zu]"
      "[embedding_dim=%zu]\n",
      ffn_fc_gb,
      s->layer_count,
      s->hidden_dim,
      s->embedding_dim
  );
  fprintf(
      f,
      "---    ffn_up (%7.4f GB) [layer_count=%zu][hidden_dim=%zu]"
      "[embedding_dim=%zu]\n",
      ffn_up_gb,
      s->layer_count,
      s->hidden_dim,
      s->embedding_dim
  );
  fprintf(
      f,
      "---   ffn_out (%7.4f GB) [layer_count=%zu][embedding_dim=%zu]"
      "[hidden_dim=%zu]\n",
      ffn_out_gb,
      s->layer_count,
      s->embedding_dim,
      s->hidden_dim
  );
  fprintf(
      f,
      "---  out_norm (%7.4f GB) [embedding_dim=%zu]\n",
      out_norm_gb,
      s->embedding_dim
  );
  if (aliased_out_weight) {
    fprintf(f, "---       out (%7.4f GB): alias to embedding\n", out_gb);
  } else {
    fprintf(
        f,
        "---       out (%7.4f GB) [vocabulary_len=%zu][embedding_dim=%zu]\n",
        out_gb,
        s->vocabulary_len,
        s->embedding_dim
    );
  }
}

// Read safetensors by parsing safetensor files in the model directory
safetensors_t* parser_parse_safetensors(const char* path);
safetensors_t* safetensors_read(options_t* options) {
  return parser_parse_safetensors(options->model_dir);
}

// Convert a string to a safetensors_type_t enum value
safetensors_type_t safetensors_type_from_string(const char* s) {
  if (!s) {
    UTIL_DIE("NULL string for safetensors_type_from_string");
  }
  size_t type_count =
      sizeof(safetensors_type_str) / sizeof(safetensors_type_str[0]);
  for (size_t i = 0; i < type_count; i++) {
    if (strcmp(s, safetensors_type_str[i]) == 0) {
      return (safetensors_type_t)i;
    }
  }
  UTIL_DIE("unknown data type");
  return 0;
}

// Lookup a file in the safetensors_t structure, adding it if not present
void safetensors_file_lookup(
    safetensors_t* safetensors, char* path, char* file
) {
  char fullpath[SAFETENSORS_MAX_STRING];
  snprintf(fullpath, sizeof(fullpath), "%s/%s", path, file);

  // Check if the file is already in the table
  for (size_t i = 0; i < safetensors->file_count; i++) {
    if (strcmp(safetensors->file[i], fullpath) == 0) {
      // File already exists, do nothing
      return;
    }
  }

  // Check capacity
  if (safetensors->file_count >= SAFETENSORS_MAX_FILE_COUNT) {
    UTIL_DIE("too many safetensors files");
  }

  // Otherwise, add it to the table
  safetensors->file[safetensors->file_count] = strdup(fullpath);
  safetensors->file_count++;
}

// Get the size in bytes of a data type
size_t safetensors_sizeof(safetensors_type_t type) {
  switch (type) {
    case SAFETENSORS_TYPE_F16:
    case SAFETENSORS_TYPE_BF16:
      return 2;
    case SAFETENSORS_TYPE_F32:
      return 4;
    default:
      UTIL_DIE("unknown safetensors type");
  }
  return 0;
}

// Return true if the output weight is aliased to the embedding weight
// If the output weight is not found, we assume it's aliased
bool safetensors_aliased_out_weight(const safetensors_t* safetensors) {
  for (size_t i = 0; i < safetensors->tensor_count; i++) {
    const safetensors_tensor_t* t = &safetensors->tensor[i];
    if (strcmp(t->name, SAFETENSORS_PATTERN_OUT_WEIGHT) == 0) {
      return false; // Found the output weight, not aliased
    }
  }
  return true;
}
