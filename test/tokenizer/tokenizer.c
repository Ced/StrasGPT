#include "tokenizer.h"
#include "options.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int json_scanner_lex_destroy(void);
int mpi_rank = 0;
int mpi_size = 1;

// Read hex-encoded UTF-8 prompts; report ids and reconstructed bytes.
int main(int argc, char** argv) {
  if (argc != 2) {
    return EXIT_FAILURE;
  }
  options_t options = {0};
  options.model_dir = argv[1];
  tokenizer_t* t = tokenizer_read(&options);
  // Every individual byte must survive binary output, including NUL and
  // bytes that are not complete UTF-8 characters on their own.
  FILE* bytes = tmpfile();
  if (!bytes) {
    return EXIT_FAILURE;
  }
  for (size_t i = 0; i < 256; i++) {
    tokenizer_print_token(bytes, t, t->byte_token[i]);
  }
  rewind(bytes);
  for (size_t i = 0; i < 256; i++) {
    if (fgetc(bytes) != (int)i) {
      return EXIT_FAILURE;
    }
  }
  if (fgetc(bytes) != EOF) {
    return EXIT_FAILURE;
  }
  fclose(bytes);
  // BOS/EOS insertion must not participate in ordinary BPE merges.
  t->bos_token_id = t->added_count ? t->added_token[0] : -1;
  t->eos_token_id = t->bos_token_id;
  size_t special_count;
  int* special;
  tokenizer_tokenize(t, "", true, true, &special_count, &special);
  if (special_count != (t->added_count ? 2u : 0u)) {
    return EXIT_FAILURE;
  }
  free(special);
  char* line = NULL;
  size_t size = 0;
  ssize_t len;
  while ((len = getline(&line, &size, stdin)) >= 0) {
    if (len && line[len - 1] == '\n') {
      line[--len] = '\0';
    }
    if (len % 2) {
      return EXIT_FAILURE;
    }
    for (ssize_t i = 0; i < len / 2; i++) {
      unsigned int byte;
      if (sscanf(line + 2 * i, "%2x", &byte) != 1 || byte == 0) {
        return EXIT_FAILURE;
      }
      line[i] = (char)byte;
    }
    line[len / 2] = '\0';
    size_t count;
    int* token;
    tokenizer_tokenize(t, line, false, false, &count, &token);
    for (size_t i = 0; i < count; i++) {
      printf("%s%d", i ? " " : "", token[i]);
    }
    printf("\n");
    for (size_t i = 0; i < count; i++) {
      for (size_t j = 0; j < t->token_string_len[token[i]]; j++) {
        printf("%02x", (unsigned char)t->token_string[token[i]][j]);
      }
    }
    printf("\n");
    free(token);
  }
  free(line);
  tokenizer_free(t);
  json_scanner_lex_destroy();
  return EXIT_SUCCESS;
}
