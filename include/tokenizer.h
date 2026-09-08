#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct options;

#define TOKENIZER_FILE             "tokenizer.json"
#define TOKENIZER_MAX_PRINT        4
#define TOKENIZER_MAX_TOKEN_STRING 262144

typedef struct {
  char* token_string;
  size_t token_string_len;
  int id;
} tokenizer_index_t;

typedef struct {
  char* string; // Temporary printable pair from JSON; freed after loading
  int left;
  int right;
  int result;
  size_t rank;
} tokenizer_merge_t;

// Supported U+2581 variants: Metaspace(first, unsplit) or Prepend + Replace.
typedef struct {
  bool enabled;
  size_t pre_tokenizer_count;
  bool replacement;
  bool first;
  bool unsplit;
  bool byte_fallback;
  size_t normalizer_type_count;
  bool normalizer_prepend;
  bool normalizer_space;
  bool normalizer_replace;
  size_t decoder_type_count;
  bool decoder_invalid;
  bool decoder_replace;
  size_t decoder_content_count;
  bool decoder_strip_start;
  bool decoder_strip_stop;
} tokenizer_metaspace_t;

typedef struct tokenizer {
  size_t token_string_count; // Largest token id + 1, including added tokens
  char* token_string[TOKENIZER_MAX_TOKEN_STRING];
  size_t token_string_len[TOKENIZER_MAX_TOKEN_STRING];
  bool added[TOKENIZER_MAX_TOKEN_STRING];
  size_t sorted_token_string_count;
  tokenizer_index_t sorted_token_string[TOKENIZER_MAX_TOKEN_STRING];
  size_t added_count;
  int added_token[TOKENIZER_MAX_TOKEN_STRING];
  size_t merge_count;
  size_t merge_capacity;
  tokenizer_merge_t* merge;
  int byte_token[256];

  tokenizer_metaspace_t metaspace;

  // Supported ByteLevel pre-tokenizer settings
  char* pattern;
  bool byte_level;
  bool ignore_merges;
  bool normalize; // Common Latin NFC compositions, not full Unicode NFC
  bool marks;
  bool case_split;
  bool contractions;
  size_t digit_count;

  int bos_token_id;
  int eos_token_id;
} tokenizer_t;

tokenizer_t* tokenizer_malloc(void);
void tokenizer_free(tokenizer_t* tokenizer);
void tokenizer_print(FILE* f, const tokenizer_t* tokenizer);
tokenizer_t* tokenizer_read(struct options* options);
void tokenizer_add_token(tokenizer_t* t, char* string, int id, bool added);
void tokenizer_add_merge(tokenizer_t* t, char* string);
void tokenizer_set_pattern(tokenizer_t* t, char* pattern);

char* tokenizer_decode(tokenizer_t* t, int token);
void tokenizer_print_token(FILE* f, tokenizer_t* t, int token);
void tokenizer_print_sequence(
    FILE* f, tokenizer_t* t, size_t token_count, int* token
);
void tokenizer_tokenize(
    tokenizer_t* t,
    char* text,
    bool bos,
    bool eos,
    size_t* token_count,
    int** token
);
void tokenizer_print_tokens(
    tokenizer_t* tokenizer,
    FILE* f,
    size_t token_count,
    int* token,
    size_t sample_count
);
#endif // TOKENIZER_H
