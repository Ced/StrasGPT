#include "options.h"
#include "tokenizer.h"
#include "util.h"
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Allocate a tokenizer_t structure
tokenizer_t* tokenizer_malloc(void) {
  tokenizer_t* tokenizer = calloc(1, sizeof(tokenizer_t));
  if (!tokenizer) {
    UTIL_DIE("failed to malloc for tokenizer_t");
  }
  return tokenizer;
}

// Free a tokenizer_t structure
void tokenizer_free(tokenizer_t* tokenizer) {
  if (tokenizer) {
    for (size_t i = 0; i < tokenizer->token_string_count; i++) {
      free(tokenizer->token_string[i]);
    }
    free(tokenizer);
  }
}

// Print a tokenizer_t structure
void tokenizer_print(FILE* f, const tokenizer_t* tokenizer) {
  if (!tokenizer) {
    fprintf(f, "Tokenizer: NULL\n");
    return;
  }

  fprintf(f, "Tokenizer:\n");
  fprintf(f, "- Special tokens:\n");
  fprintf(f, "--- bos_token_id: %d\n", tokenizer->bos_token_id);
  fprintf(f, "--- eos_token_id: %d\n", tokenizer->eos_token_id);

  // Print token strings (only first and last TOKENIZER_MAX_PRINT)
  size_t sample_count = TOKENIZER_MAX_PRINT;
  size_t total = tokenizer->token_string_count;

  fprintf(f, "- Token strings (%zu):\n", total);
  size_t i = 0;
  for (; i < sample_count && i < total; i++) {
    fprintf(
        f,
        "--- Token string[%6zu]: score=%10.3f string=%s\n",
        i,
        tokenizer->score[i],
        tokenizer->token_string[i]
    );
  }
  if (2 * sample_count < total) {
    fprintf(stderr, "--- ...\n");
  }
  size_t tail_start = (2 * sample_count < total) ? (total - sample_count) : i;
  for (i = tail_start; i < total; i++) {
    fprintf(
        f,
        "--- Token string[%6zu]: score=%10.3f string=%s\n",
        i,
        tokenizer->score[i],
        tokenizer->token_string[i]
    );
  }

  fprintf(f, "- Max token string length: %zu\n", tokenizer->max_token_string_len);
}

static int compare_token_strings(const void* a, const void* b) {
  return strcmp(
      ((tokenizer_index_t*)a)->token_string,
      ((tokenizer_index_t*)b)->token_string
  );
}

// Decode byte-level BPE string in-place
// Strings from the tokenizer.json file have been processed by tiktoken that
// uses a reversible "byte -> printable Unicode" map so every byte 0–255 can
// appear in a UTF-8 string (e.g., space character " " becomes readable "Ġ").
// The following function decodes strings back, this is done in-place as
// decoded string is shorter than encoded one.
static void decode_bpe_bytes_inplace(char* str) {
  size_t read_pos = 0;
  size_t write_pos = 0;
  size_t len = strlen(str);

  while (read_pos < len) {
    unsigned char c = (unsigned char)str[read_pos];

    // Check if this is a 2-byte UTF-8 sequence
    if (c >= 0xC0 && c <= 0xDF && read_pos + 1 < len) {
      unsigned char c2 = (unsigned char)str[read_pos + 1];
      int codepoint = ((c & 0x1F) << 6) | (c2 & 0x3F);

      // Check if this is in the byte-level BPE range (U+0100 to U+01FF)
      if (codepoint >= 0x0100 && codepoint <= 0x01FF) {
        str[write_pos++] = (char)(codepoint - 0x0100);
        read_pos += 2;
      } else {
        // Not a byte-level encoding, keep as-is
        str[write_pos++] = str[read_pos++];
      }
    } else {
      // Regular ASCII or already decoded
      str[write_pos++] = str[read_pos++];
    }
  }

  str[write_pos] = '\0';
}

// Read and prepare a tokenizer from options, extracted from tokenizer.json
// file in the model directory. Token strings are decoded in-place and sorted
// by score. Single-byte strings are filled.
tokenizer_t* parser_parse_tokenizer(const char* path);
tokenizer_t* tokenizer_read(options_t* options) {
  // Read vocabulary (strings + scores)
  tokenizer_t* tokenizer = parser_parse_tokenizer(options->model_dir);

  // Decode "printable unicode" tiktoken strings to raw bytes
  for (size_t i = 0; i < tokenizer->token_string_count; i++) {
    decode_bpe_bytes_inplace(tokenizer->token_string[i]);
  }

  // Sort according to scores
  for (size_t i = 0; i < tokenizer->token_string_count; i++) {
    tokenizer->sorted_token_string[i].token_string = tokenizer->token_string[i];
    tokenizer->sorted_token_string[i].id = i;
  }
  qsort(
      tokenizer->sorted_token_string,
      tokenizer->token_string_count,
      sizeof(*tokenizer->sorted_token_string),
      compare_token_strings
  );

  // Fill single-byte strings
  for (size_t i = 0; i < TOKENIZER_MAX_BYTE_STRING / 2; i++) {
    tokenizer->byte_string[i * 2] = (unsigned char)i;
    tokenizer->byte_string[i * 2 + 1] = '\0';
  }

  // Set max_token_string_len
  size_t max = 0;
  for (size_t i = 0; i < tokenizer->token_string_count; i++) {
    size_t len = strlen(tokenizer->token_string[i]);
    max = (len > max) ? len : max;
  }
  tokenizer->max_token_string_len = max;

  return tokenizer;
}

// ----------------------------------------------------------------------------
// Note: remainder of the code is mostly from llama2.c and llama3.c projects

// Print a token string, taking care of raw byte tokens
void tokenizer_print_token_string(FILE* f, char* token_string) {
  if (token_string == NULL) {
    return;
  }
  if (token_string[0] == '\0') {
    return;
  }
  if (token_string[1] == '\0') {
    unsigned char byte_val = token_string[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // Bad byte, don't print it
    }
  }
  fprintf(f, "%s", token_string);
  fflush(f);
}

// Decode a token into a string, taking care of some special tokens and raw
// byte tokens
char* tokenizer_decode(tokenizer_t* t, int token) {
  if (token == t->bos_token_id) {
    return TOKENIZER_STRING_TOKEN_BOS;
  } else if (token == t->eos_token_id) {
    return TOKENIZER_STRING_TOKEN_EOS;
  }

  // If token is out of range, we assume special token
  if (token < 0 || (size_t)token >= t->token_string_count) {
    return "<|unknown_token|>";
  }

  char* token_string = t->token_string[token];

  // Careful, some token designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(token_string, "<0x%02hhX>", &byte_val) == 1) {
    token_string = (char*)t->byte_string + byte_val * 2;
  }
  return token_string;
}

// Efficiently look up a string in the sorted token strings, return its index
static int str_lookup(
    char* str,
    tokenizer_index_t* sorted_token_string,
    int token_string_count
) {
  // Efficiently find the perfect match for str in vocab, return its index or -1
  // if not found
  tokenizer_index_t tok = {.token_string = str}; // Acts as the key to search for
  tokenizer_index_t* res = bsearch(
      &tok,
      sorted_token_string,
      token_string_count,
      sizeof(*sorted_token_string),
      compare_token_strings
  );
  return res != NULL ? res->id : -1;
}

// Tokenize a text string into an array of tokens
// bos != 0 means prepend the BOS token, eos != 0 means append the EOS token
void tokenizer_tokenize(
    tokenizer_t* t,
    char* text,
    bool bos,
    bool eos,
    size_t* token_count,
    int** token_ptr
) {
  if (text == NULL) {
    UTIL_DIE("cannot encode NULL text");
  }

  // Allocate +3 for '\0', ?BOS, ?EOS
  *token_ptr = malloc((strlen(text) + 3) * sizeof(**token_ptr));
  if (*token_ptr == NULL) {
    UTIL_DIE("failed to malloc for tokens");
  }
  int* token = *token_ptr;

  // Create a temporary buffer that will store merge candidates of always two
  // consecutive token *2 for concat, +1 for null terminator +2 for UTF8 (in
  // case max_token_length is 1)
  size_t str_buffer_size = (t->max_token_string_len * 2 + 1 + 2) * sizeof(char);
  char* str_buffer = malloc(str_buffer_size);
  size_t str_len = 0;

  // Start at 0 token
  *token_count = 0;

  // Add optional BOS token, if desired
  if (bos) {
    token[(*token_count)++] = t->bos_token_id;
  }

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have
  // the energy to read more of the sentencepiece code to figure out what it's
  // doing

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000  U+007F    0xxxxxxx
  // U+0080  U+07FF    110xxxxx 10xxxxxx
  // U+0800  U+FFFF    1110xxxx 10xxxxxx 10xxxxxx
  // U+10000 U+10FFFF  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

  // Process the raw (UTF-8) byte sequence of the input string
  for (char* c = text; *c != '\0'; c++) {

    // Reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the
    // rest 0x80 is 10000000 in UTF-8, all continuation bytes start with "10" in
    // first two bits so in English this is: "if this byte is not a continuation
    // byte"
    if ((*c & 0xC0) != 0x80) {
      // This byte must be either a leading byte (11...) or an ASCII char
      // (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // Append the current byte to the buffer
    // note: ++ is post-increment, incremented after this line
    str_buffer[str_len++] = *c;
    str_buffer[str_len] = '\0';

    // While the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning
    // str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // OK c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(
        str_buffer, t->sorted_token_string, t->token_string_count
    );

    if (id != -1) {
      // We found this codepoint in vocab, add it as a token
      token[(*token_count)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (size_t i = 0; i < str_len; i++) {
        token[(*token_count)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // Protect against a sequence of stray UTF8 continuation bytes
  }

  // Merge the best consecutive pair or triple each iteration, according to the
  // scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    // Length of the best merge sequence (2 for pair, 3 for triple)
    int best_len = 2;

    // First, try to find the best pair to merge
    for (int i = 0; i < ((int)*token_count - 1); i++) {
      // Check if we can merge the pair (token[i], token[i+1])
      snprintf(
          str_buffer,
          str_buffer_size,
          "%s%s",
          t->token_string[token[i]],
          t->token_string[token[i + 1]]
      );
      int id = str_lookup(
          str_buffer, t->sorted_token_string, t->token_string_count
      );
      if (id != -1 && t->score[id] > best_score) {
        // This merge pair exists in vocab! record its score and position
        best_score = t->score[id];
        best_id = id;
        best_idx = i;
      }
    }

    // If no pair was found, try to find the best triple to merge
    if (best_idx == -1) {
      for (int i = 0; i < ((int)*token_count - 2); i++) {
        // Check if we can merge the triple (token[i], token[i+1],
        // token[i+2])
        snprintf(
            str_buffer,
            str_buffer_size,
            "%s%s%s",
            t->token_string[token[i]],
            t->token_string[token[i + 1]],
            t->token_string[token[i + 2]]
        );
        int id = str_lookup(
            str_buffer, t->sorted_token_string, t->token_string_count
        );
        if (id != -1 && t->score[id] > best_score) {
          // This merge triple exists in vocab! record its score and position
          best_score = t->score[id];
          best_id = id;
          best_idx = i;
          best_len = 3;
        }
      }
    }

    if (best_idx == -1) {
      // We couldn't find any more pairs or triples to merge, so we're done
      break;
    }

    // Merge the consecutive pair or triple (best_idx, best_idx+1[, best_idx+2])
    // into new token best_id
    token[best_idx] = best_id;
    // Delete token(s) at position best_idx+1 (and optionally best_idx+2), shift
    // the entire sequence back
    for (int i = best_idx + 1; i < ((int)*token_count - best_len + 1); i++) {
      token[i] = token[i + best_len - 1];
    }
    // Token length decreased by the number of merged token minus one
    (*token_count) -= (best_len - 1);
  }

  // Add optional EOS token, if desired
  if (eos) {
    token[(*token_count)++] = t->eos_token_id;
  }

  free(str_buffer);
}

// Print the first and last sample_count tokens from a token array
void tokenizer_print_tokens(
    tokenizer_t* tokenizer,
    FILE* f,
    size_t token_count,
    int* token,
    size_t sample_count
) {
  fprintf(f, "Tokens (%zu):\n", token_count);
  size_t i = 0;
  for (; i < sample_count && i < token_count; i++) {
    fprintf(f, "- Token[%4zu]: %6d (\"", i, token[i]);
    char* token_string = tokenizer_decode(tokenizer, token[i]);
    tokenizer_print_token_string(f, token_string);
    fprintf(f, "\")\n");
  }
  if (2 * sample_count < token_count) {
    fprintf(f, "- ...\n");
  }
  size_t tail_start =
      (2 * sample_count < token_count) ? (token_count - sample_count) : i;
  for (i = tail_start; i < token_count; i++) {
    fprintf(f, "- Token[%4zu]: %6d (\"", i, token[i]);
    char* token_string = tokenizer_decode(tokenizer, token[i]);
    tokenizer_print_token_string(f, token_string);
    fprintf(f, "\")\n");
  }
  fflush(f);
}
