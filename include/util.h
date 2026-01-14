#ifndef UTIL_H
# define UTIL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// For errors that should never happen
#define UTIL_DIE(msg) \
    do { \
        fprintf( \
            stderr, \
            "[StrasGPT] Error at %s:%d (%s): %s\n", \
            __FILE__, \
            __LINE__, \
            __func__, \
            msg); \
        exit(EXIT_FAILURE); \
    } while(0)

// For basic errors (bad file name, bad argument, etc.)
#define UTIL_ERROR(msg) \
    do { \
        fprintf(stderr, "[StrasGPT] Error: %s\n", msg); \
        exit(EXIT_FAILURE); \
    } while(0)

#define UTIL_ALIGNMENT 32
#define UTIL_GIGA      (1024 * 1024 * 1024)
#define UTIL_MIN(a,b)  (((a)<(b))?(a):(b))
#define UTIL_MAX(a,b)  (((a)>(b))?(a):(b))

static inline float util_bf16_to_f32(uint16_t w) {
  union {
    uint32_t u;
    float f;
  } u = {(uint32_t)w << 16};
  return u.f;
}

typedef enum {
    UTIL_MATRIX_TYPE_INT8,
    UTIL_MATRIX_TYPE_BF16,
    UTIL_MATRIX_TYPE_FP32
} util_matrix_type_t;

void util_matrix_generic_print(
    size_t row_count,
    size_t col_count,
    size_t row_sample_count,
    size_t col_sample_count,
    void* m,
    util_matrix_type_t type
);

#define util_matrix_print( \
    row_count, col_count, row_sample_count, col_sample_count, m) \
    _Generic((m), \
        int8_t*: util_matrix_generic_print( \
            row_count, \
            col_count, \
            row_sample_count, \
            col_sample_count, \
            (void*)m, \
            UTIL_MATRIX_TYPE_INT8), \
        uint16_t*: util_matrix_generic_print( \
            row_count, \
            col_count, \
            row_sample_count, \
            col_sample_count, \
            (void*)m, \
            UTIL_MATRIX_TYPE_BF16), \
        float*: util_matrix_generic_print( \
            row_count, \
            col_count, \
            row_sample_count, \
            col_sample_count, \
            (void*)m, \
            UTIL_MATRIX_TYPE_FP32) \
    )

void util_matrix_summary_fp32(
    const char* name,
    size_t row_count,
    size_t col_count,
    size_t sample_count,
    const float* m
);

void util_matrix_summary_bf16(
    const char* name,
    size_t row_count,
    size_t col_count,
    size_t sample_count,
    const uint16_t* m
);

void util_matrix_summary_int8(
    const char* name,
    size_t row_count,
    size_t col_count,
    size_t sample_count,
    const int8_t* m
);

#define util_matrix_summary(name, row_count, col_count, sample_count, m) \
    _Generic((m), \
        float*: util_matrix_summary_fp32, \
        uint16_t*: util_matrix_summary_bf16, \
        int8_t*: util_matrix_summary_int8 \
    )(name, row_count, col_count, sample_count, m)

void util_parse_tokens(
    char* input,
    size_t* token_count,
    int** tokens,
    bool add_bos,
    int bos_token_id
);

#endif // UTIL_H
