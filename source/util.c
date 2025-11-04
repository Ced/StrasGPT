#include "util.h"
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

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
