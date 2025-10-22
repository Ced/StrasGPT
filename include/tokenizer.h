#ifndef TOKENIZER_H
# define TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

struct options;

#define TOKENIZER_FILE             "tokenizer.json"
#define TOKENIZER_MAX_PRINT        4
#define TOKENIZER_MAX_TOKEN_STRING 262144 // Enough for TikToken
#define TOKENIZER_MAX_BYTE_STRING  512    // Must be multiple of 2
#define TOKENIZER_STRING_TOKEN_BOS "<|begin_of_text|>"
#define TOKENIZER_STRING_TOKEN_EOS "<|end_of_text|>"

typedef struct {
  char* token_string;
  int id;
} tokenizer_index_t;

typedef struct tokenizer {
  size_t token_string_count;
  char* token_string[TOKENIZER_MAX_TOKEN_STRING];
  float score[TOKENIZER_MAX_TOKEN_STRING];
  tokenizer_index_t sorted_token_string[TOKENIZER_MAX_TOKEN_STRING];
  size_t max_token_string_len;
  unsigned char byte_string[TOKENIZER_MAX_BYTE_STRING]; // Single-byte strings

  // Special tokens
  int bos_token_id; // Beginning of string token id
  int eos_token_id; // End of string token id
} tokenizer_t;

tokenizer_t* tokenizer_malloc(void);
void tokenizer_free(tokenizer_t* tokenizer);
void tokenizer_print(FILE* f, const tokenizer_t* tokenizer);
tokenizer_t* tokenizer_read(struct options* options);

void tokenizer_print_token_string(FILE* f, char* token_string);
char* tokenizer_decode(tokenizer_t* t, int token);
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
