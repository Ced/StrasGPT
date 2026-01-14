#include "util.h"
#include <stdbool.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------------

// matrix_print:
// ---------------------
// Pretty-prints a linearized matrix. The matrix is displayed row by row,
// printing only first and last few rows and columns.
// This operation supports int8_t, _Float16 and float types via _Generic.
//
// Parameters:
// - row_count:        Number of rows in the matrix
// - col_count:        Number of columns in the matrix
// - row_sample_count: Number of rows to print from the beginning and end.
// - col_sample_count: Number of columns to print from the beginning and end
// - m:                Pointer to the linearized matrix

// Helper to print a single row
static void row_generic_print(
  size_t row_index,
  size_t col_count,
  size_t col_sample_count,
  void* m,
  util_matrix_type_t type
) {
  printf("[");
  int overlap = (col_sample_count * 2 >= col_count);
  for (size_t j = 0; j < (overlap ? col_count : col_sample_count); ++j) {
    size_t idx = row_index * col_count + j;
    switch (type) {
      case UTIL_MATRIX_TYPE_INT8:
        printf(" %4d", ((int8_t*)m)[idx]);
        break;
      case UTIL_MATRIX_TYPE_BF16:
        printf(" %6.3f", util_bf16_to_f32(((uint16_t*)m)[idx]));
        break;
      case UTIL_MATRIX_TYPE_FP32:
        printf(" %6.3f", ((float*)m)[idx]);
        break;
    }
  }

  // Ellipsis if there's a gap
  if (!overlap) {
    printf(" ...");
    // Last part
    for (size_t j = col_count - col_sample_count; j < col_count; ++j) {
      size_t idx = row_index * col_count + j;
      switch (type) {
        case UTIL_MATRIX_TYPE_INT8:
          printf(" %4d", ((int8_t*)m)[idx]);
          break;
        case UTIL_MATRIX_TYPE_BF16:
          printf(" %6.3f", util_bf16_to_f32(((uint16_t*)m)[idx]));
          break;
        case UTIL_MATRIX_TYPE_FP32:
          printf(" %6.3f", ((float*)m)[idx]);
          break;
      }
    }
  }
  printf(" ]\n");
}

void util_matrix_generic_print(
    size_t row_count,
    size_t col_count,
    size_t row_sample_count,
    size_t col_sample_count,
    void* m,
    util_matrix_type_t type
) {
  // Print first sample rows
  for (size_t i = 0; i < row_sample_count && i < row_count; ++i) {
    row_generic_print(i, col_count, col_sample_count, m, type);
  }

  // Print ellipsis if needed
  if (row_sample_count * 2 < row_count) {
    printf("...\n");
  }

  // Print last sample rows
  for (size_t i = row_count - row_sample_count; i < row_count; ++i) {
    if (i >= row_sample_count) { // Avoid double-printing if samples overlap
      row_generic_print(i, col_count, col_sample_count, m, type);
    }
  }
}

// ----------------------------------------------------------------------------

// matrix_summary:
// ---------------
// Prints matrix summary to stderr: selected samples (first and last N values),
// min, max, and sum. Supports int8_t, _Float16 and float types via _Generic.
//
// Parameters:
// - name:         Matrix name (string)
// - row_count:    Number of rows in the matrix
// - col_count:    Number of columns in the matrix
// - sample_count: Number of values to show at the beginning and end
// - m:            Pointer to matrix buffer

void util_matrix_summary_bf16(
    const char* name,
    size_t row_count,
    size_t col_count,
    size_t sample_count,
    const uint16_t* m
) {
  size_t total = row_count * col_count;
  if (total == 0) {
    fprintf(stderr, "%s: empty matrix\n", name ? name : "matrix");
    return;
  }

  double first = (double)util_bf16_to_f32(m[0]);
  double min = first;
  double max = first;
  double sum = 0.0;

  fprintf(stderr, "%9s: [", name ? name : "matrix");
  size_t i = 0;
  for (; i < sample_count && i < total; i++) {
    fprintf(stderr, " %6.3f", (double)util_bf16_to_f32(m[i]));
  }
  if (2 * sample_count < total) {
    fprintf(stderr, " ...");
  }
  size_t tail_start = (2 * sample_count < total) ? (total - sample_count) : i;
  for (i = tail_start; i < total; i++) {
    fprintf(stderr, " %6.3f", (double)util_bf16_to_f32(m[i]));
  }
  fprintf(stderr, " ] ");

  for (i = 0; i < total; ++i) {
    double val = (double)util_bf16_to_f32(m[i]);
    if (val < min) {
      min = val;
    }
    if (val > max) {
      max = val;
    }
    sum += val;
  }
  double mean = sum / (double)total;
  fprintf(stderr, "min=%6.3f max=%6.3f ", min, max);
  fprintf(stderr, "mean=%6.3f sum=%6.3f\n", mean, sum);
}

#define DEFINE_UTIL_MATRIX_SUMMARY(TYPE, NAME) \
    void NAME( \
      const char* name, \
      size_t row_count, \
      size_t col_count, \
      size_t sample_count, \
      const TYPE* m \
    ) { \
      size_t total = row_count * col_count; \
      if (total == 0) { \
        fprintf(stderr, "%9s: empty matrix\n", name ? name : "matrix"); \
        return; \
      } \
      double min = (double)m[0]; \
      double max = (double)m[0]; \
      double sum = 0.0f; \
      fprintf(stderr, "%6s: [", name ? name : "matrix"); \
      size_t i; \
      for (i = 0; i < sample_count && i < total; i++) { \
        fprintf(stderr, "%6.3f ", (double)m[i]); \
      } \
      if (2 * sample_count < total) { \
        fprintf(stderr, "... "); \
      } \
      size_t tail_start = (2 * sample_count < total) ? total - sample_count : i; \
      for (i = tail_start; i < total; i++) { \
        fprintf(stderr, "%6.3f ", (double)m[i]); \
      } \
      fprintf(stderr, "] "); \
      for (i = 0; i < total; i++) { \
        double val = (double)m[i]; \
        if (val < min) { \
          min = val; \
        } \
        if (val > max) { \
          max = val; \
        } \
        sum += val; \
      } \
      double mean = sum / (double)total; \
      fprintf(stderr, "min=%8.3f max=%8.3f ", min, max); \
      fprintf(stderr, "mean=%8.3f sum=%8.3f\n", mean, sum); \
    }

DEFINE_UTIL_MATRIX_SUMMARY(float, util_matrix_summary_fp32)
DEFINE_UTIL_MATRIX_SUMMARY(int8_t, util_matrix_summary_int8)

// ----------------------------------------------------------------------------

// util_parse_tokens:
// ------------------
// Parse a space-separated or comma-separated string of token IDs into an integer array.
// The input string should contain whitespace and/or comma-separated integer token IDs.
//
// Parameters:
// - input:        Input string containing space/comma-separated token IDs
// - token_count:  Pointer to store the number of tokens parsed
// - tokens:       Pointer to store the allocated array of token IDs
// - add_bos:      If true, prepend BOS token to the token array
// - bos_token_id: Token ID to use as BOS if add_bos is true
//
// Returns:
// - void (exits on error via UTIL_DIE/UTIL_ERROR)
//
// Notes:
// - The caller is responsible for freeing the allocated tokens array
// - Accepts space, tab, newline, carriage return, and comma as separators

void util_parse_tokens(
    char* input,
    size_t* token_count,
    int** tokens,
    bool add_bos,
    int bos_token_id
) {
  if (input == NULL) {
    UTIL_DIE("cannot parse NULL input");
  }

  // Make a copy of the input string for tokenization
  char* input_copy = strdup(input);
  if (!input_copy) {
    UTIL_DIE("malloc failed for input_copy");
  }

  // First pass: count tokens
  size_t parsed_count = 0;
  char* saveptr;
  char* tok = strtok_r(input_copy, " \t\n\r,", &saveptr);
  while (tok != NULL) {
    parsed_count++;
    tok = strtok_r(NULL, " \t\n\r,", &saveptr);
  }

  if (parsed_count < 1) {
    free(input_copy);
    UTIL_ERROR("expected at least 1 token in pre-tokenized input");
  }

  // Calculate final token count (including optional BOS)
  *token_count = parsed_count + (add_bos ? 1 : 0);

  // Allocate token array
  *tokens = malloc(*token_count * sizeof(int));
  if (!*tokens) {
    free(input_copy);
    UTIL_DIE("malloc failed for tokens");
  }

  // Optionally add BOS token at the beginning
  size_t idx = 0;
  if (add_bos) {
    (*tokens)[idx++] = bos_token_id;
  }

  // Make a fresh copy for second pass
  free(input_copy);
  input_copy = strdup(input);
  if (!input_copy) {
    free(*tokens);
    UTIL_DIE("malloc failed for input_copy");
  }

  // Second pass: parse and store tokens
  tok = strtok_r(input_copy, " \t\n\r,", &saveptr);
  while (tok != NULL && idx < *token_count) {
    char* endptr;
    long token_id = strtol(tok, &endptr, 10);
    if (*endptr != '\0') {
      free(input_copy);
      free(*tokens);
      fprintf(stderr, "[StrasGPT] Error: invalid token ID: %s\n", tok);
      exit(EXIT_FAILURE);
    }
    (*tokens)[idx++] = (int)token_id;
    tok = strtok_r(NULL, " \t\n\r,", &saveptr);
  }

  free(input_copy);
}
