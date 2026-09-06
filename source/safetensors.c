#include "safetensors.h"
#include "options.h"
#include "util.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* safetensors_type_str[] = {
    "F16", "BF16", "F32", "U8", "MXFP4", "E8M0"
};

// Allocate a safetensors_t structure
safetensors_t* safetensors_malloc(void) {
  safetensors_t* safetensors = calloc(1, sizeof(safetensors_t));
  if (!safetensors) {
    UTIL_DIE("failed to malloc for safetensors_t");
  }
  safetensors->swa_len = 0;
  safetensors->rope_yarn = false;
  safetensors->rope_factor = 1.0f;
  safetensors->rope_context_len = 0;
  safetensors->rope_beta_fast = 32.0f;
  safetensors->rope_beta_slow = 1.0f;
  safetensors->rope_yarn_truncate = true;

  safetensors->expert_count = 0;
  safetensors->expert_per_token_count = 0;
  safetensors->ffn_swiglu_limit = 0.0f;
  for (size_t i = 0; i < SAFETENSORS_MAX_LAYER_COUNT; i++) {
    safetensors->layer_type[i] = SAFETENSORS_LAYER_TYPE_FA;
  }
  return safetensors;
}

static const char* safetensors_layer_type_string(
    safetensors_layer_type_t layer_type
) {
  switch (layer_type) {
    case SAFETENSORS_LAYER_TYPE_SWA:
      return "SWA"; // "sliding_window_attention"
    case SAFETENSORS_LAYER_TYPE_LA:
      return "LA";  // "linear_attention"
    case SAFETENSORS_LAYER_TYPE_FA:
    default:
      return "FA"; // "full_attention"
  }
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
    fprintf(f, "--- model_type:             %s\n", safetensors->model_type);
  } else {
    fprintf(f, "--- model_type:             (null)\n");
  }
  fprintf(f, "--- embedding_dim:          %zu\n", safetensors->embedding_dim);
  fprintf(f, "--- head_dim:               %zu\n", safetensors->head_dim);
  fprintf(f, "--- hidden_dim:             %zu\n", safetensors->hidden_dim);
  fprintf(f, "--- layer_count:            %zu\n", safetensors->layer_count);
  fprintf(f, "--- layer_types:            ");
  safetensors_print_layer_types(f, safetensors);
  fprintf(f, "--- q_head_count:           %zu\n", safetensors->q_head_count);
  fprintf(f, "--- kv_head_count:          %zu\n", safetensors->kv_head_count);
  fprintf(
      f,
      "--- mha_output_gate:        %s\n",
      safetensors->mha_output_gate ? "true" : "false"
  );
  fprintf(f, "--- la_kernel_size:         %zu\n", safetensors->la_kernel_size);
  fprintf(f, "--- la_k_head_dim:          %zu\n", safetensors->la_k_head_dim);
  fprintf(f, "--- la_k_head_count:        %zu\n", safetensors->la_k_head_count);
  fprintf(f, "--- la_v_head_dim:          %zu\n", safetensors->la_v_head_dim);
  fprintf(f, "--- la_v_head_count:        %zu\n", safetensors->la_v_head_count);
  fprintf(f, "--- vocabulary_len:         %zu\n", safetensors->vocabulary_len);
  fprintf(f, "--- context_len:            %zu\n", safetensors->context_len);
  fprintf(f, "--- swa_len:                %zu\n", safetensors->swa_len);
  fprintf(f, "--- expert_count:           %zu\n", safetensors->expert_count);
  fprintf(
      f,
      "--- expert_per_token_count: %zu\n",
      safetensors->expert_per_token_count
  );
  fprintf(f, "--- ffn_swiglu_limit:       %g\n", safetensors->ffn_swiglu_limit);
  fprintf(f, "--- rope_theta:             %.1f\n", safetensors->rope_theta);
  fprintf(
      f,
      "--- rope_yarn:              %s\n",
      safetensors->rope_yarn ? "true" : "false"
  );
  fprintf(f, "--- rope_factor:            %g\n", safetensors->rope_factor);
  fprintf(
      f, "--- rope_context_len:       %zu\n", safetensors->rope_context_len
  );
  fprintf(f, "--- rope_beta_fast:         %g\n", safetensors->rope_beta_fast);
  fprintf(f, "--- rope_beta_slow:         %g\n", safetensors->rope_beta_slow);
  fprintf(
      f,
      "--- rope_yarn_truncate:     %s\n",
      safetensors->rope_yarn_truncate ? "true" : "false"
  );
  fprintf(
      f,
      "--- rope_interleaved:       %s\n",
      safetensors->rope_interleaved ? "true" : "false"
  );
  fprintf(f, "--- mrope_sections:         ");
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
  fprintf(f, "--- bos_token_id:           %d\n", safetensors->bos_token_id);
  fprintf(f, "--- eos_token_id:           %d\n", safetensors->eos_token_id);

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

// Get tensor storage size according to name pattern (adds all sizes
// of layer tensors witht he same suffix).
static size_t tensor_size(const safetensors_t* s, const char* pattern) {
  const char* index = strstr(pattern, "%d");
  const char* suffix = index ? index + 2 : pattern;
  size_t bytes = 0;
  size_t suffix_len = strlen(suffix);
  for (size_t i = 0; i < s->tensor_count; i++) {
    const safetensors_tensor_t* t = &s->tensor[i];
    size_t name_len = strlen(t->name);
    if (name_len >= suffix_len &&
        strcmp(t->name + name_len - suffix_len, suffix) == 0) {
      bytes += t->size;
    }
  }
  return bytes;
}

// Print a safetensors_t structure
void safetensors_print_model_infos(FILE* f, const safetensors_t* s) {
  if (!s) {
    fprintf(f, "model: NULL\n");
    return;
  }

  size_t q_head_per_kv_head_count = s->q_head_count / s->kv_head_count;
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
  fprintf(f, "--- swa_len:                  %zu\n", s->swa_len);
  fprintf(f, "--- expert_count:             %zu\n", s->expert_count);
  fprintf(f, "--- expert_per_token_count:   %zu\n", s->expert_per_token_count);
  fprintf(f, "--- ffn_swiglu_limit:         %g\n", s->ffn_swiglu_limit);
  fprintf(f, "--- epsilon:                  %.g\n", s->epsilon);
  fprintf(f, "--- rope_theta:               %.1f\n", s->rope_theta);
  fprintf(
      f, "--- rope_yarn:                %s\n", s->rope_yarn ? "true" : "false"
  );
  fprintf(f, "--- rope_factor:              %g\n", s->rope_factor);
  fprintf(f, "--- rope_context_len:         %zu\n", s->rope_context_len);
  fprintf(f, "--- rope_beta_fast:           %g\n", s->rope_beta_fast);
  fprintf(f, "--- rope_beta_slow:           %g\n", s->rope_beta_slow);
  fprintf(
      f,
      "--- rope_yarn_truncate:       %s\n",
      s->rope_yarn_truncate ? "true" : "false"
  );
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

  size_t la_qkv_dim = 2 * s->la_k_head_count * s->la_k_head_dim +
                      s->la_v_head_count * s->la_v_head_dim;
  size_t la_v_dim = s->la_v_head_count * s->la_v_head_dim;

  // clang-format off
  double gb = UTIL_GIGA;
  double embedding_gb =
      tensor_size(s, SAFETENSORS_PATTERN_EMBEDDING_WEIGHT) / gb;
  double mha_norm_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_NORM_WEIGHT) / gb;
  double mha_q_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_Q_WEIGHT) / gb;
  // Gated attention stores query and gate rows in the same tensor.
  if (mha_output_gate) {
    mha_q_gb /= 2;
  }
  double mha_gate_gb = mha_output_gate ? mha_q_gb : 0;
  double mha_q_norm_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_Q_NORM_WEIGHT) / gb;
  double mha_k_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_K_WEIGHT) / gb;
  double mha_k_norm_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_K_NORM_WEIGHT) / gb;
  double mha_v_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_V_WEIGHT) / gb;
  double mha_out_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_OUT_WEIGHT) / gb;
  double la_qkv_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_QKV_WEIGHT) / gb;
  double la_gate_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_GATE_WEIGHT) / gb;
  double la_alpha_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_ALPHA_WEIGHT) / gb;
  double la_beta_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_BETA_WEIGHT) / gb;
  double la_dt_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_DT_BIAS) / gb;
  double la_decay_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_DECAY_WEIGHT) / gb;
  double la_conv_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_CONV_WEIGHT) / gb;
  double la_norm_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_NORM_WEIGHT) / gb;
  double la_out_gb =
      tensor_size(s, SAFETENSORS_PATTERN_LA_OUT_WEIGHT) / gb;
  double ffn_norm_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_NORM_WEIGHT) / gb;
  double ffn_fc_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_FC_WEIGHT) / gb;
  double ffn_up_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_UP_WEIGHT) / gb;
  double ffn_out_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_OUT_WEIGHT) / gb;
  double out_norm_gb =
      tensor_size(s, SAFETENSORS_PATTERN_OUT_NORM_WEIGHT) / gb;
  double out_gb =
      tensor_size(s, SAFETENSORS_PATTERN_OUT_WEIGHT) / gb;

  // Split combined gate/up tensors equally between their two arrays.
  double mha_q_bias_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_Q_BIAS) / gb;
  double mha_k_bias_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_K_BIAS) / gb;
  double mha_v_bias_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_V_BIAS) / gb;
  double mha_out_bias_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_OUT_BIAS) / gb;
  double mha_sinks_gb =
      tensor_size(s, SAFETENSORS_PATTERN_MHA_SINKS) / gb;
  double ffn_router_weight_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_ROUTER_WEIGHT) / gb;
  double ffn_router_bias_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_ROUTER_BIAS) / gb;
  double ffn_xp_gate_block_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_GATE_UP_BLOCK) / (2 * gb);
  double ffn_xp_gate_scale_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_GATE_UP_SCALE) / (2 * gb);
  double ffn_xp_gate_bias_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_GATE_UP_BIAS) / (2 * gb);
  double ffn_xp_up_block_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_GATE_UP_BLOCK) / (2 * gb);
  double ffn_xp_up_scale_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_GATE_UP_SCALE) / (2 * gb);
  double ffn_xp_up_bias_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_GATE_UP_BIAS) / (2 * gb);
  double ffn_xp_down_block_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_DOWN_BLOCK) / gb;
  double ffn_xp_down_scale_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_DOWN_SCALE) / gb;
  double ffn_xp_down_bias_gb =
      tensor_size(s, SAFETENSORS_PATTERN_FFN_XP_DOWN_BIAS) / gb;
  // clang-format on

  double total_gb = 0.0;
  total_gb += embedding_gb;
  total_gb += mha_norm_gb;
  total_gb += mha_q_gb;
  total_gb += mha_gate_gb;
  total_gb += mha_q_norm_gb;
  total_gb += mha_k_gb;
  total_gb += mha_k_norm_gb;
  total_gb += mha_v_gb;
  total_gb += mha_out_gb;
  total_gb += mha_q_bias_gb;
  total_gb += mha_k_bias_gb;
  total_gb += mha_v_bias_gb;
  total_gb += mha_out_bias_gb;
  total_gb += mha_sinks_gb;

  total_gb += la_qkv_gb;
  total_gb += la_gate_gb;
  total_gb += la_alpha_gb;
  total_gb += la_beta_gb;
  total_gb += la_dt_gb;
  total_gb += la_decay_gb;
  total_gb += la_conv_gb;
  total_gb += la_norm_gb;
  total_gb += la_out_gb;

  total_gb += ffn_norm_gb;
  total_gb += ffn_fc_gb;
  total_gb += ffn_up_gb;
  total_gb += ffn_out_gb;
  total_gb += ffn_router_weight_gb;
  total_gb += ffn_router_bias_gb;
  total_gb += ffn_xp_gate_block_gb;
  total_gb += ffn_xp_gate_scale_gb;
  total_gb += ffn_xp_gate_bias_gb;
  total_gb += ffn_xp_up_block_gb;
  total_gb += ffn_xp_up_scale_gb;
  total_gb += ffn_xp_up_bias_gb;
  total_gb += ffn_xp_down_block_gb;
  total_gb += ffn_xp_down_scale_gb;
  total_gb += ffn_xp_down_bias_gb;

  total_gb += out_norm_gb;
  total_gb += out_gb;

  double non_layer_gb = embedding_gb + out_norm_gb + out_gb;
  double per_layer_gb = (total_gb - non_layer_gb) / s->layer_count;

  // Tensor names use a fixed width of 17 characters.
  fprintf(
      f,
      "- Tensors (total %.2f GB, non-layer %.2f GB, per-layer %.2f GB):\n",
      total_gb,
      non_layer_gb,
      per_layer_gb
  );
  fprintf(
      f,
      "---  embedding_weight (%7.4f GB) BF16  "
      "[vocabulary_len=%zu][embedding_dim=%zu]\n",
      embedding_gb,
      s->vocabulary_len,
      s->embedding_dim
  );
  fprintf(
      f,
      "---   mha_norm_weight (%7.4f GB) BF16  "
      "[layer_count=%zu][embedding_dim=%zu]\n",
      mha_norm_gb,
      s->layer_count,
      s->embedding_dim
  );
  if (fa_layer_count > 0) {
    fprintf(
        f,
        "---      mha_q_weight (%7.4f GB) BF16  "
        "[fa_layer_count=%zu][kv_head_count=%zu]"
        "[q_head_per_kv_head_count=%zu][head_dim=%zu][embedding_dim=%zu]\n",
        mha_q_gb,
        fa_layer_count,
        s->kv_head_count,
        q_head_per_kv_head_count,
        s->head_dim,
        s->embedding_dim
    );
    if (mha_q_bias_gb > 0) {
      fprintf(
          f,
          "---        mha_q_bias (%7.4f GB) BF16  "
          "[fa_layer_count=%zu][kv_head_count=%zu]"
          "[q_head_per_kv_head_count=%zu][head_dim=%zu]\n",
          mha_q_bias_gb,
          fa_layer_count,
          s->kv_head_count,
          q_head_per_kv_head_count,
          s->head_dim
      );
    }
    if (mha_output_gate) {
      fprintf(
          f,
          "---   mha_gate_weight (%7.4f GB) BF16  "
          "[fa_layer_count=%zu][kv_head_count=%zu]"
          "[q_head_per_kv_head_count=%zu][head_dim=%zu][embedding_dim=%zu]\n",
          mha_gate_gb,
          fa_layer_count,
          s->kv_head_count,
          q_head_per_kv_head_count,
          s->head_dim,
          s->embedding_dim
      );
    }
    if (mha_q_norm_gb > 0) {
      fprintf(
          f,
          "--- mha_q_norm_weight (%7.4f GB) BF16  "
          "[fa_layer_count=%zu][head_dim=%zu]\n",
          mha_q_norm_gb,
          fa_layer_count,
          s->head_dim
      );
    }
    fprintf(
        f,
        "---      mha_k_weight (%7.4f GB) BF16  "
        "[fa_layer_count=%zu][kv_head_count=%zu]"
        "[head_dim=%zu][embedding_dim=%zu]\n",
        mha_k_gb,
        fa_layer_count,
        s->kv_head_count,
        s->head_dim,
        s->embedding_dim
    );
    if (mha_k_bias_gb > 0) {
      fprintf(
          f,
          "---        mha_k_bias (%7.4f GB) BF16  "
          "[fa_layer_count=%zu][kv_head_count=%zu][head_dim=%zu]\n",
          mha_k_bias_gb,
          fa_layer_count,
          s->kv_head_count,
          s->head_dim
      );
    }
    if (mha_k_norm_gb > 0) {
      fprintf(
          f,
          "--- mha_k_norm_weight (%7.4f GB) BF16  "
          "[fa_layer_count=%zu][head_dim=%zu]\n",
          mha_k_norm_gb,
          fa_layer_count,
          s->head_dim
      );
    }
    fprintf(
        f,
        "---      mha_v_weight (%7.4f GB) BF16  "
        "[fa_layer_count=%zu][kv_head_count=%zu]"
        "[head_dim=%zu][embedding_dim=%zu]\n",
        mha_v_gb,
        fa_layer_count,
        s->kv_head_count,
        s->head_dim,
        s->embedding_dim
    );
    if (mha_v_bias_gb > 0) {
      fprintf(
          f,
          "---        mha_v_bias (%7.4f GB) BF16  "
          "[fa_layer_count=%zu][kv_head_count=%zu][head_dim=%zu]\n",
          mha_v_bias_gb,
          fa_layer_count,
          s->kv_head_count,
          s->head_dim
      );
    }
    fprintf(
        f,
        "---    mha_out_weight (%7.4f GB) BF16  "
        "[fa_layer_count=%zu][embedding_dim=%zu]"
        "[q_head_count*head_dim=%zu]\n",
        mha_out_gb,
        fa_layer_count,
        s->embedding_dim,
        s->q_head_count * s->head_dim
    );
    if (mha_out_bias_gb > 0) {
      fprintf(
          f,
          "---      mha_out_bias (%7.4f GB) BF16  "
          "[fa_layer_count=%zu][embedding_dim=%zu]\n",
          mha_out_bias_gb,
          fa_layer_count,
          s->embedding_dim
      );
    }
    if (mha_sinks_gb > 0) {
      fprintf(
          f,
          "---         mha_sinks (%7.4f GB) BF16  "
          "[fa_layer_count=%zu][q_head_count=%zu]\n",
          mha_sinks_gb,
          fa_layer_count,
          s->q_head_count
      );
    }
  }
  if (la_layer_count > 0) {
    fprintf(
        f,
        "---     la_qkv_weight (%7.4f GB) BF16  "
        "[la_layer_count=%zu][la_qkv_dim=%zu]"
        "[embedding_dim=%zu]\n",
        la_qkv_gb,
        la_layer_count,
        la_qkv_dim,
        s->embedding_dim
    );
    fprintf(
        f,
        "---    la_gate_weight (%7.4f GB) BF16  "
        "[la_layer_count=%zu][la_v_dim=%zu]"
        "[embedding_dim=%zu]\n",
        la_gate_gb,
        la_layer_count,
        la_v_dim,
        s->embedding_dim
    );
    fprintf(
        f,
        "---   la_alpha_weight (%7.4f GB) BF16  "
        "[la_layer_count=%zu][la_v_head_count=%zu]"
        "[embedding_dim=%zu]\n",
        la_alpha_gb,
        la_layer_count,
        s->la_v_head_count,
        s->embedding_dim
    );
    fprintf(
        f,
        "---    la_beta_weight (%7.4f GB) BF16  "
        "[la_layer_count=%zu][la_v_head_count=%zu]"
        "[embedding_dim=%zu]\n",
        la_beta_gb,
        la_layer_count,
        s->la_v_head_count,
        s->embedding_dim
    );
    fprintf(
        f,
        "---        la_dt_bias (%7.4f GB) BF16  "
        "[la_layer_count=%zu]"
        "[la_v_head_count=%zu]\n",
        la_dt_gb,
        la_layer_count,
        s->la_v_head_count
    );
    fprintf(
        f,
        "---   la_decay_weight (%7.4f GB) F32   "
        "[la_layer_count=%zu]"
        "[la_v_head_count=%zu] (-exp(A_log))\n",
        la_decay_gb,
        la_layer_count,
        s->la_v_head_count
    );
    fprintf(
        f,
        "---    la_conv_weight (%7.4f GB) BF16  "
        "[la_layer_count=%zu][la_qkv_dim=%zu]"
        "[la_kernel_size=%zu]\n",
        la_conv_gb,
        la_layer_count,
        la_qkv_dim,
        s->la_kernel_size
    );
    fprintf(
        f,
        "---    la_norm_weight (%7.4f GB) F32   "
        "[la_layer_count=%zu]"
        "[la_v_head_dim=%zu]\n",
        la_norm_gb,
        la_layer_count,
        s->la_v_head_dim
    );
    fprintf(
        f,
        "---     la_out_weight (%7.4f GB) BF16  "
        "[la_layer_count=%zu][embedding_dim=%zu]"
        "[la_v_dim=%zu]\n",
        la_out_gb,
        la_layer_count,
        s->embedding_dim,
        la_v_dim
    );
  }
  fprintf(
      f,
      "---   ffn_norm_weight (%7.4f GB) BF16  "
      "[layer_count=%zu][embedding_dim=%zu]\n",
      ffn_norm_gb,
      s->layer_count,
      s->embedding_dim
  );
  if (s->expert_count == 0) {
    fprintf(
        f,
        "---     ffn_fc_weight (%7.4f GB) BF16  "
        "[layer_count=%zu][hidden_dim=%zu]"
        "[embedding_dim=%zu]\n",
        ffn_fc_gb,
        s->layer_count,
        s->hidden_dim,
        s->embedding_dim
    );
    fprintf(
        f,
        "---     ffn_up_weight (%7.4f GB) BF16  "
        "[layer_count=%zu][hidden_dim=%zu]"
        "[embedding_dim=%zu]\n",
        ffn_up_gb,
        s->layer_count,
        s->hidden_dim,
        s->embedding_dim
    );
    fprintf(
        f,
        "---    ffn_out_weight (%7.4f GB) BF16  "
        "[layer_count=%zu][embedding_dim=%zu]"
        "[hidden_dim=%zu]\n",
        ffn_out_gb,
        s->layer_count,
        s->embedding_dim,
        s->hidden_dim
    );
  }

  if (ffn_router_weight_gb > 0) {
    fprintf(
        f,
        "--- ffn_router_weight (%7.4f GB) BF16  "
        "[layer_count=%zu][expert_count=%zu][embedding_dim=%zu]\n",
        ffn_router_weight_gb,
        s->layer_count,
        s->expert_count,
        s->embedding_dim
    );
  }
  if (ffn_router_bias_gb > 0) {
    fprintf(
        f,
        "---   ffn_router_bias (%7.4f GB) BF16  "
        "[layer_count=%zu][expert_count=%zu]\n",
        ffn_router_bias_gb,
        s->layer_count,
        s->expert_count
    );
  }
  if (ffn_xp_gate_block_gb > 0) {
    fprintf(
        f,
        "--- ffn_xp_gate_block (%7.4f GB) MXFP4 "
        "[layer_count=%zu][expert_count=%zu][hidden_dim=%zu]"
        "[embedding_dim/32=%zu][16]\n",
        ffn_xp_gate_block_gb,
        s->layer_count,
        s->expert_count,
        s->hidden_dim,
        s->embedding_dim / 32
    );
  }
  if (ffn_xp_gate_scale_gb > 0) {
    fprintf(
        f,
        "--- ffn_xp_gate_scale (%7.4f GB) E8M0  "
        "[layer_count=%zu][expert_count=%zu][hidden_dim=%zu]"
        "[embedding_dim/32=%zu]\n",
        ffn_xp_gate_scale_gb,
        s->layer_count,
        s->expert_count,
        s->hidden_dim,
        s->embedding_dim / 32
    );
  }
  if (ffn_xp_gate_bias_gb > 0) {
    fprintf(
        f,
        "---  ffn_xp_gate_bias (%7.4f GB) BF16  "
        "[layer_count=%zu][expert_count=%zu][hidden_dim=%zu]\n",
        ffn_xp_gate_bias_gb,
        s->layer_count,
        s->expert_count,
        s->hidden_dim
    );
  }
  if (ffn_xp_up_block_gb > 0) {
    fprintf(
        f,
        "---   ffn_xp_up_block (%7.4f GB) MXFP4 "
        "[layer_count=%zu][expert_count=%zu][hidden_dim=%zu]"
        "[embedding_dim/32=%zu][16]\n",
        ffn_xp_up_block_gb,
        s->layer_count,
        s->expert_count,
        s->hidden_dim,
        s->embedding_dim / 32
    );
  }
  if (ffn_xp_up_scale_gb > 0) {
    fprintf(
        f,
        "---   ffn_xp_up_scale (%7.4f GB) E8M0  "
        "[layer_count=%zu][expert_count=%zu][hidden_dim=%zu]"
        "[embedding_dim/32=%zu]\n",
        ffn_xp_up_scale_gb,
        s->layer_count,
        s->expert_count,
        s->hidden_dim,
        s->embedding_dim / 32
    );
  }
  if (ffn_xp_up_bias_gb > 0) {
    fprintf(
        f,
        "---    ffn_xp_up_bias (%7.4f GB) BF16  "
        "[layer_count=%zu][expert_count=%zu][hidden_dim=%zu]\n",
        ffn_xp_up_bias_gb,
        s->layer_count,
        s->expert_count,
        s->hidden_dim
    );
  }
  if (ffn_xp_down_block_gb > 0) {
    fprintf(
        f,
        "--- ffn_xp_down_block (%7.4f GB) MXFP4 "
        "[layer_count=%zu][expert_count=%zu][embedding_dim=%zu]"
        "[hidden_dim/32=%zu][16]\n",
        ffn_xp_down_block_gb,
        s->layer_count,
        s->expert_count,
        s->embedding_dim,
        s->hidden_dim / 32
    );
  }
  if (ffn_xp_down_scale_gb > 0) {
    fprintf(
        f,
        "--- ffn_xp_down_scale (%7.4f GB) E8M0  "
        "[layer_count=%zu][expert_count=%zu][embedding_dim=%zu]"
        "[hidden_dim/32=%zu]\n",
        ffn_xp_down_scale_gb,
        s->layer_count,
        s->expert_count,
        s->embedding_dim,
        s->hidden_dim / 32
    );
  }
  if (ffn_xp_down_bias_gb > 0) {
    fprintf(
        f,
        "---  ffn_xp_down_bias (%7.4f GB) BF16  "
        "[layer_count=%zu][expert_count=%zu][embedding_dim=%zu]\n",
        ffn_xp_down_bias_gb,
        s->layer_count,
        s->expert_count,
        s->embedding_dim
    );
  }
  fprintf(
      f,
      "---   out_norm_weight (%7.4f GB) BF16  "
      "[embedding_dim=%zu]\n",
      out_norm_gb,
      s->embedding_dim
  );
  if (aliased_out_weight) {
    fprintf(
        f,
        "---        out_weight (%7.4f GB) BF16  : "
        "alias to embedding_weight\n",
        out_gb
    );
  } else {
    fprintf(
        f,
        "---        out_weight (%7.4f GB) BF16  "
        "[vocabulary_len=%zu][embedding_dim=%zu]\n",
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

// Recognize GPT-OSS expert projection storage without regex configuration.
static safetensors_type_t gpt_oss_byte_type(const char* name) {
  const char* prefix = "model.layers.";
  if (!name || strncmp(name, prefix, strlen(prefix)) != 0) {
    return SAFETENSORS_TYPE_U8;
  }
  const char* suffix = name + strlen(prefix);
  if (*suffix < '0' || *suffix > '9') {
    return SAFETENSORS_TYPE_U8;
  }
  while (*suffix >= '0' && *suffix <= '9') {
    suffix++;
  }
  if (strcmp(suffix, ".mlp.experts.gate_up_proj_blocks") == 0 ||
      strcmp(suffix, ".mlp.experts.down_proj_blocks") == 0) {
    return SAFETENSORS_TYPE_MXFP4;
  }
  if (strcmp(suffix, ".mlp.experts.gate_up_proj_scales") == 0 ||
      strcmp(suffix, ".mlp.experts.down_proj_scales") == 0) {
    return SAFETENSORS_TYPE_E8M0;
  }
  return SAFETENSORS_TYPE_U8;
}

// Convert a stored dtype, interpreting known GPT-OSS byte tensors.
safetensors_type_t safetensors_type_from_string(
    const char* s, const char* model_type, const char* tensor_name
) {
  if (!s) {
    UTIL_DIE("NULL string for safetensors_type_from_string");
  }
  if (strcmp(s, "U8") == 0 && model_type &&
      strcmp(model_type, "gpt_oss") == 0) {
    return gpt_oss_byte_type(tensor_name);
  }
  // MXFP4 and E8M0 are logical types, not checkpoint dtype strings.
  size_t type_count = SAFETENSORS_TYPE_U8 + 1;
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

// Bytes per stored element; MXFP4 stores two logical values per byte.
size_t safetensors_sizeof(safetensors_type_t type) {
  switch (type) {
    case SAFETENSORS_TYPE_F16:
    case SAFETENSORS_TYPE_BF16:
      return 2;
    case SAFETENSORS_TYPE_F32:
      return 4;
    case SAFETENSORS_TYPE_U8:
    case SAFETENSORS_TYPE_MXFP4:
    case SAFETENSORS_TYPE_E8M0:
      return 1;
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
