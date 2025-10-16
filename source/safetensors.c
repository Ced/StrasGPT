#include "options.h"
#include "safetensors.h"
#include "util.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* safetensors_type_str[] = {
  "F16",
  "BF16",
  "F32"
};

// Allocate a safetensors_t structure
safetensors_t* safetensors_malloc(void) {
  safetensors_t* safetensors = calloc(1, sizeof(safetensors_t));
  if (!safetensors) {
    UTIL_DIE("failed to malloc for safetensors_t");
  }
  return safetensors;
}

// Free a safetensors_t structure
void safetensors_free(safetensors_t* safetensors) {
  if (safetensors) {
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
  fprintf(f, "--- embedding_dim:  %zu\n", safetensors->embedding_dim);
  fprintf(f, "--- hidden_dim:     %zu\n", safetensors->hidden_dim);
  fprintf(f, "--- layer_count:    %zu\n", safetensors->layer_count);
  fprintf(f, "--- q_head_count:   %zu\n", safetensors->q_head_count);
  fprintf(f, "--- kv_head_count:  %zu\n", safetensors->kv_head_count);
  fprintf(f, "--- vocabulary_len: %zu\n", safetensors->vocabulary_len);
  fprintf(f, "--- context_len:    %zu\n", safetensors->context_len);

  fprintf(f, "- Files (%zu):\n", safetensors->file_count);
  for (size_t i = 0; i < safetensors->file_count; i++) {
    fprintf(f, "--- File[%-zu]: %s\n", i, safetensors->file[i]);
  }

  fprintf(f, "- Tensors (%zu):\n", safetensors->tensor_count);
  // Let's prepare to align the output
  size_t max_id_len = 0;
  size_t max_name_len = 0;
  size_t max_type_len =  0;
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

// Read safetensors by parsing safetensor files in the model directory
safetensors_t* parser_parse_safetensors(const char* path);
safetensors_t* safetensors_read(options_t* options) {
  return parser_parse_safetensors(options->model_dir);
}

// Convert a string to a safetensors_type_t enum value
safetensors_type_t safetensors_type_from_string(const char *s) {
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
