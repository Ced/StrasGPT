// Simple, self-contained BPE with ByteLevel and Metaspace input paths.

#include "options.h"
#include "tokenizer.h"
#include "util.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

tokenizer_t* tokenizer_malloc(void) {
  tokenizer_t* t = calloc(1, sizeof(*t));
  if (!t) {
    UTIL_DIE("failed to malloc for tokenizer");
  }
  // Missing special-token IDs are absent, not token zero.
  t->bos_token_id = -1;
  t->eos_token_id = -1;
  return t;
}

void tokenizer_free(tokenizer_t* t) {
  if (!t) {
    return;
  }
  for (size_t i = 0; i < t->token_string_count; i++) {
    free(t->token_string[i]);
  }
  for (size_t i = 0; i < t->merge_count; i++) {
    free(t->merge[i].string);
  }
  if (t->metaspace.enabled) {
    for (size_t i = 0; i < t->sorted_token_string_count; i++) {
      free(t->sorted_token_string[i].token_string);
    }
  }
  free(t->merge);
  free(t->pattern);
  free(t);
}

void tokenizer_add_token(tokenizer_t* t, char* string, int id, bool added) {
  if (id < 0 || id >= TOKENIZER_MAX_TOKEN_STRING || !string[0]) {
    UTIL_ERROR("invalid tokenizer vocabulary entry");
  }
  if (t->token_string[id]) {
    // Added tokens may also appear in the model vocabulary.
    if (strcmp(t->token_string[id], string) || (!added && !t->added[id])) {
      UTIL_ERROR("duplicate tokenizer vocabulary id");
    }
    free(string);
  } else {
    t->token_string[id] = string;
  }
  t->added[id] |= added;
  t->token_string_count = UTIL_MAX(t->token_string_count, (size_t)id + 1);
}

// Merge priority follows model.merges order, independently of token IDs.
void tokenizer_add_merge(tokenizer_t* t, char* string) {
  if (t->merge_count == t->merge_capacity) {
    t->merge_capacity = t->merge_capacity ? t->merge_capacity * 2 : 1024;
    tokenizer_merge_t* merge =
        realloc(t->merge, t->merge_capacity * sizeof(*merge));
    if (!merge) {
      UTIL_DIE("failed to realloc for tokenizer merges");
    }
    t->merge = merge;
  }
  t->merge[t->merge_count] =
      (tokenizer_merge_t){.string = string, .rank = t->merge_count};
  t->merge_count++;
}

// Read one valid UTF-8 codepoint.
static unsigned int utf8_read(const char** text) {
  const unsigned char* p = (const unsigned char*)*text;
  unsigned int cp = *p++;
  size_t count = 0;
  unsigned int min = 0;
  if (cp >= 0xc2 && cp <= 0xdf) {
    cp &= 31;
    count = 1;
    min = 0x80;
  } else if (cp >= 0xe0 && cp <= 0xef) {
    cp &= 15;
    count = 2;
    min = 0x800;
  } else if (cp >= 0xf0 && cp <= 0xf4) {
    cp &= 7;
    count = 3;
    min = 0x10000;
  } else if (cp >= 0x80) {
    UTIL_ERROR("invalid UTF-8 in tokenizer input");
  }
  for (size_t i = 0; i < count; i++) {
    if ((*p & 0xc0) != 0x80) {
      UTIL_ERROR("invalid UTF-8 continuation");
    }
    cp = (cp << 6) | (*p++ & 63);
  }
  if (cp < min || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
    UTIL_ERROR("invalid UTF-8 codepoint");
  }
  *text = (const char*)p;
  return cp;
}

// Inverse of the GPT-2/Hugging Face ByteLevel alphabet. Printable Latin-1
// bytes keep their codepoints; the other bytes use U+0100 and up in order.
// JSON escapes have already been decoded by the parser. The returned byte
// length includes any embedded NUL, so callers must not use strlen instead.
static size_t decode_bpe_bytes(char* string) {
  int byte[512];
  for (size_t i = 0; i < 512; i++) {
    byte[i] = -1;
  }
  size_t next = 256;
  for (size_t i = 0; i < 256; i++) {
    bool printable =
        (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174;
    byte[printable ? i : next++] = (int)i;
  }
  const char* in = string;
  size_t len = 0;
  while (*in) {
    unsigned int cp = utf8_read(&in);
    if (cp >= 512 || byte[cp] < 0) {
      UTIL_ERROR("invalid ByteLevel vocabulary character");
    }
    string[len++] = (char)byte[cp];
  }
  string[len] = '\0';
  return len;
}

static int compare_token_strings(const void* a, const void* b) {
  const tokenizer_index_t* x = a;
  const tokenizer_index_t* y = b;
  size_t len = UTIL_MIN(x->token_string_len, y->token_string_len);
  int cmp = memcmp(x->token_string, y->token_string, len);
  if (cmp) {
    return cmp;
  }
  return (x->token_string_len > y->token_string_len) -
         (x->token_string_len < y->token_string_len);
}

static int str_lookup(tokenizer_t* t, char* string, size_t len) {
  tokenizer_index_t key = {.token_string = string, .token_string_len = len};
  tokenizer_index_t* found = bsearch(
      &key,
      t->sorted_token_string,
      t->sorted_token_string_count,
      sizeof(key),
      compare_token_strings
  );
  return found ? found->id : -1;
}

static int compare_merges(const void* a, const void* b) {
  const tokenizer_merge_t* x = a;
  const tokenizer_merge_t* y = b;
  if (x->left != y->left) {
    return (x->left > y->left) ? 1 : -1;
  }
  return (x->right > y->right) - (x->right < y->right);
}

// Recognize the Split + ByteLevel patterns used by Qwen3, Qwen3.5, Llama 3,
// Mistral Nemo and GPT-OSS. Select rules from the pattern, not the model name.
// Matching uses loops with no regex dependency; unknown patterns are rejected.
void tokenizer_set_pattern(tokenizer_t* t, char* pattern) {
  static const char pattern_qwen3[] =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{"
      "L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+"
      "(?!\\S)|\\s+";
  static const char pattern_qwen35[] =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p"
      "{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s"
      "*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  static const char pattern_llama[] =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{"
      "L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]"
      "+|\\s+(?!\\S)|\\s+";
  static const char pattern_nemo[] =
      "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
      "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{L"
      "u}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M"
      "}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s"
      "+(?!\\S)|\\s+";
  static const char pattern_gpt_oss[] =
      "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
      "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll"
      "|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\"
      "p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|"
      "'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|"
      "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  const char* patterns[] = {
      pattern_qwen3,
      pattern_qwen35,
      pattern_llama,
      pattern_nemo,
      pattern_gpt_oss
  };
  if (t->pattern) {
    UTIL_ERROR("multiple tokenizer split patterns");
  }
  for (size_t i = 0; i < 5; i++) {
    if (strcmp(pattern, patterns[i])) {
      continue;
    }
    t->pattern = pattern;
    t->marks = i == 1 || i >= 3;
    t->case_split = i >= 3;
    t->contractions = i != 3;
    t->digit_count = (i == 2 || i == 4) ? 3 : 1;
    return;
  }
  UTIL_ERROR("unsupported tokenizer split pattern");
}

// Keep raw spellings in the lookup index, and decoded bytes in token_string.
// This separation preserves the distinction between 'a' and '<0x61>'.
static void metaspace_prepare(tokenizer_t* t) {
  tokenizer_metaspace_t* m = &t->metaspace;
  if (t->byte_level || t->pattern || t->normalize || t->ignore_merges ||
      !m->byte_fallback ||
      m->decoder_invalid || m->decoder_type_count != 5 || !m->decoder_replace ||
      m->decoder_content_count != 2 || !m->decoder_strip_start ||
      !m->decoder_strip_stop) {
    UTIL_ERROR("unsupported Metaspace tokenizer configuration");
  }
  if (m->normalizer_type_count) {
    if (m->pre_tokenizer_count || m->normalizer_type_count != 3 ||
        !m->normalizer_prepend || !m->normalizer_space ||
        !m->normalizer_replace) {
      UTIL_ERROR("expected Prepend + Replace with no pre-tokenizer");
    }
  } else if (m->pre_tokenizer_count != 1 || !m->replacement || !m->first ||
             !m->unsplit) {
    UTIL_ERROR("expected Metaspace with prepend=first and split=false");
  }
  if (str_lookup(t, "\xe2\x96\x81", 3) < 0) {
    UTIL_ERROR("missing Metaspace space token");
  }
  for (size_t i = 0; i < 256; i++) {
    char string[7];
    snprintf(string, sizeof(string), "<0x%02X>", (unsigned int)i);
    t->byte_token[i] = str_lookup(t, string, 6);
    if (t->byte_token[i] < 0) {
      UTIL_ERROR("missing Metaspace byte fallback token");
    }
  }
  for (size_t i = 0; i < t->sorted_token_string_count; i++) {
    int id = t->sorted_token_string[i].id;
    const char* in = t->token_string[id];
    char* decoded = malloc(strlen(in) + 1);
    if (!decoded) {
      UTIL_DIE("failed to malloc for Metaspace decoded token");
    }
    size_t len = 0;
    while (*in) {
      if (!strncmp(in, "\xe2\x96\x81", 3)) {
        decoded[len++] = ' ';
        in += 3;
      } else {
        decoded[len++] = *in++;
      }
    }
    decoded[len] = '\0';
    t->token_string[id] = decoded;
    t->token_string_len[id] = len;
  }
  for (size_t i = 0; i < 256; i++) {
    int id = t->byte_token[i];
    t->token_string[id][0] = (char)i;
    t->token_string[id][1] = '\0';
    t->token_string_len[id] = 1;
  }
}

tokenizer_t* parser_parse_tokenizer(const char* path);
// Load the supported tokenizer.json subset, not arbitrary HF pipelines.
// The parser rejects unsupported normalizer types and added-token options.
// Scans, arrays and binary searches keep the implementation straightforward.
tokenizer_t* tokenizer_read(options_t* options) {
  tokenizer_t* t = parser_parse_tokenizer(options->model_dir);
  if (!t->metaspace.enabled && (!t->byte_level || !t->pattern)) {
    UTIL_ERROR("expected a supported Split + ByteLevel tokenizer");
  }
  for (size_t i = 0; i < 256; i++) {
    t->byte_token[i] = -1;
  }
  for (size_t i = 0; i < t->token_string_count; i++) {
    char* string = t->token_string[i];
    if (!string) {
      continue;
    }
    size_t len;
    if (t->added[i]) {
      t->added_token[t->added_count++] = (int)i;
      len = strlen(string);
    } else {
      len = t->metaspace.enabled ? strlen(string) : decode_bpe_bytes(string);
      if (!t->metaspace.enabled && len == 1) {
        t->byte_token[(unsigned char)string[0]] = (int)i;
      }
      t->sorted_token_string[t->sorted_token_string_count++] =
          (tokenizer_index_t){string, len, (int)i};
    }
    t->token_string_len[i] = len;
  }
  for (size_t i = 0; i < 256; i++) {
    if (!t->metaspace.enabled && t->byte_token[i] < 0) {
      UTIL_ERROR("missing ByteLevel byte token");
    }
  }
  qsort(
      t->sorted_token_string,
      t->sorted_token_string_count,
      sizeof(*t->sorted_token_string),
      compare_token_strings
  );
  for (size_t i = 0; i < t->merge_count; i++) {
    tokenizer_merge_t* m = t->merge + i;
    char* right = strchr(m->string, ' ');
    if (!right || right == m->string || !right[1] || strchr(right + 1, ' ')) {
      UTIL_ERROR("invalid tokenizer merge pair");
    }
    *right++ = '\0';
    size_t left_len = t->metaspace.enabled ? strlen(m->string)
                                          : decode_bpe_bytes(m->string);
    size_t right_len = t->metaspace.enabled ? strlen(right)
                                           : decode_bpe_bytes(right);
    m->left = str_lookup(t, m->string, left_len);
    m->right = str_lookup(t, right, right_len);
    memmove(m->string + left_len, right, right_len);
    m->result = str_lookup(t, m->string, left_len + right_len);
    if (m->left < 0 || m->right < 0 || m->result < 0) {
      UTIL_ERROR("merge references missing vocabulary entry");
    }
    free(m->string);
    m->string = NULL;
  }
  qsort(t->merge, t->merge_count, sizeof(*t->merge), compare_merges);
  for (size_t i = 1; i < t->merge_count; i++) {
    if (!compare_merges(t->merge + i - 1, t->merge + i)) {
      UTIL_ERROR("duplicate tokenizer merge pair");
    }
  }
  if (t->metaspace.enabled) {
    metaspace_prepare(t);
  }
  return t;
}

// Return actual token spellings, including added tokens and special tokens.
char* tokenizer_decode(tokenizer_t* t, int token) {
  if (token < 0 || (size_t)token >= t->token_string_count ||
      !t->token_string[token]) {
    return "<|unknown_token|>";
  }
  return t->token_string[token];
}

// Tokens can end in the middle of a UTF-8 character, or contain NUL.
// Preserve every byte rather than filter incomplete characters.
void tokenizer_print_token(FILE* f, tokenizer_t* t, int token) {
  char* string = tokenizer_decode(t, token);
  size_t len = strlen(string);
  if (token >= 0 && (size_t)token < t->token_string_count &&
      t->token_string[token]) {
    len = t->token_string_len[token];
  }
  fwrite(string, 1, len, f);
  fflush(f);
}

// Strip exactly one leading space from a complete Metaspace sequence.
// Individual token printing preserves bytes for incremental generation.
void tokenizer_print_sequence(
    FILE* f, tokenizer_t* t, size_t token_count, int* token
) {
  for (size_t i = 0; i < token_count; i++) {
    char* string = tokenizer_decode(t, token[i]);
    if (t->metaspace.enabled && i == 0 && string[0] == ' ') {
      fwrite(string + 1, 1, t->token_string_len[token[i]] - 1, f);
    } else {
      tokenizer_print_token(f, t, token[i]);
    }
  }
  fflush(f);
}

void tokenizer_print(FILE* f, const tokenizer_t* t) {
  if (!t) {
    fprintf(f, "Tokenizer: NULL\n");
    return;
  }
  fprintf(f, "Tokenizer:\n");
  fprintf(f, "- Token id range: %zu\n", t->token_string_count);
  fprintf(f, "- Merge count: %zu\n", t->merge_count);
  fprintf(f, "- Added token count: %zu\n", t->added_count);
  fprintf(f, "- BOS token id: %d\n", t->bos_token_id);
  fprintf(f, "- EOS token id: %d\n", t->eos_token_id);
}

void tokenizer_print_tokens(
    tokenizer_t* t, FILE* f, size_t token_count, int* token, size_t sample_count
) {
  fprintf(f, "Tokens (%zu):\n", token_count);
  for (size_t i = 0; i < token_count; i++) {
    if (i >= sample_count && token_count - i > sample_count) {
      fprintf(f, "- ...\n");
      i = token_count - sample_count - 1;
      continue;
    }
    fprintf(f, "- Token[%4zu]: %6d (\"", i, token[i]);
    tokenizer_print_token(f, t, token[i]);
    fprintf(f, "\")\n");
  }
  fflush(f);
}

// Classes cover ASCII, Latin-1, French ligatures, marks U+0300..U+036F and
// Unicode whitespace, including ordinary and narrow nonbreaking spaces.
// Other characters use the punctuation class: their bytes are preserved,
// but token IDs outside this coverage can differ from the reference.
enum {
  TOKENIZER_UPPER = 1,
  TOKENIZER_LOWER = 2,
  TOKENIZER_OTHER = 4, // Uncased letters
  TOKENIZER_MARK = 8,
  TOKENIZER_NUMBER = 16,
  TOKENIZER_SPACE = 32,
  TOKENIZER_LETTER = TOKENIZER_UPPER | TOKENIZER_LOWER | TOKENIZER_OTHER
};

static unsigned int character_class(unsigned int cp) {
  if ((cp >= 'A' && cp <= 'Z') || (cp >= 0xc0 && cp <= 0xde && cp != 0xd7) ||
      cp == 0x152 || cp == 0x178) {
    return TOKENIZER_UPPER;
  }
  if ((cp >= 'a' && cp <= 'z') || (cp >= 0xdf && cp <= 0xff && cp != 0xf7) ||
      cp == 0x153 || cp == 0xb5) {
    return TOKENIZER_LOWER;
  }
  if (cp == 0xaa || cp == 0xba) {
    return TOKENIZER_OTHER;
  }
  if ((cp >= '0' && cp <= '9') || cp == 0xb2 || cp == 0xb3 || cp == 0xb9 ||
      (cp >= 0xbc && cp <= 0xbe)) {
    return TOKENIZER_NUMBER;
  }
  if (cp >= 0x300 && cp <= 0x36f) {
    return TOKENIZER_MARK;
  }
  if ((cp >= 9 && cp <= 13) || cp == 32 || cp == 0x85 || cp == 0xa0 ||
      cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
      cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000) {
    return TOKENIZER_SPACE;
  }
  return 0;
}

// For models requesting NFC, compose common Latin-1 base/accent pairs and
// Y with diaeresis. This covers common decomposed French accents, not full
// Unicode NFC, and needs no Unicode library.
static unsigned int compose_latin(unsigned int base, unsigned int mark) {
  static const unsigned int composition[][3] = {
      {0x0041, 0x0300, 0x00c0}, {0x0041, 0x0301, 0x00c1},
      {0x0041, 0x0302, 0x00c2}, {0x0041, 0x0303, 0x00c3},
      {0x0041, 0x0308, 0x00c4}, {0x0041, 0x030a, 0x00c5},
      {0x0043, 0x0327, 0x00c7}, {0x0045, 0x0300, 0x00c8},
      {0x0045, 0x0301, 0x00c9}, {0x0045, 0x0302, 0x00ca},
      {0x0045, 0x0308, 0x00cb}, {0x0049, 0x0300, 0x00cc},
      {0x0049, 0x0301, 0x00cd}, {0x0049, 0x0302, 0x00ce},
      {0x0049, 0x0308, 0x00cf}, {0x004e, 0x0303, 0x00d1},
      {0x004f, 0x0300, 0x00d2}, {0x004f, 0x0301, 0x00d3},
      {0x004f, 0x0302, 0x00d4}, {0x004f, 0x0303, 0x00d5},
      {0x004f, 0x0308, 0x00d6}, {0x0055, 0x0300, 0x00d9},
      {0x0055, 0x0301, 0x00da}, {0x0055, 0x0302, 0x00db},
      {0x0055, 0x0308, 0x00dc}, {0x0059, 0x0301, 0x00dd},
      {0x0061, 0x0300, 0x00e0}, {0x0061, 0x0301, 0x00e1},
      {0x0061, 0x0302, 0x00e2}, {0x0061, 0x0303, 0x00e3},
      {0x0061, 0x0308, 0x00e4}, {0x0061, 0x030a, 0x00e5},
      {0x0063, 0x0327, 0x00e7}, {0x0065, 0x0300, 0x00e8},
      {0x0065, 0x0301, 0x00e9}, {0x0065, 0x0302, 0x00ea},
      {0x0065, 0x0308, 0x00eb}, {0x0069, 0x0300, 0x00ec},
      {0x0069, 0x0301, 0x00ed}, {0x0069, 0x0302, 0x00ee},
      {0x0069, 0x0308, 0x00ef}, {0x006e, 0x0303, 0x00f1},
      {0x006f, 0x0300, 0x00f2}, {0x006f, 0x0301, 0x00f3},
      {0x006f, 0x0302, 0x00f4}, {0x006f, 0x0303, 0x00f5},
      {0x006f, 0x0308, 0x00f6}, {0x0075, 0x0300, 0x00f9},
      {0x0075, 0x0301, 0x00fa}, {0x0075, 0x0302, 0x00fb},
      {0x0075, 0x0308, 0x00fc}, {0x0079, 0x0301, 0x00fd},
      {0x0079, 0x0308, 0x00ff}, {0x0059, 0x0308, 0x0178},
  };
  size_t count = sizeof(composition) / sizeof(*composition);
  for (size_t i = 0; i < count; i++) {
    if (composition[i][0] == base && composition[i][1] == mark) {
      return composition[i][2];
    }
  }
  return 0;
}

static size_t contraction_len(const unsigned int* cp, size_t len, size_t i) {
  if (i + 1 >= len || cp[i] != '\'') {
    return 0;
  }
  unsigned int a = cp[i + 1];
  if (a >= 'A' && a <= 'Z') {
    a += 'a' - 'A';
  }
  if (a == 's' || a == 't' || a == 'm' || a == 'd') {
    return 2;
  }
  if (i + 2 >= len) {
    return 0;
  }
  unsigned int b = cp[i + 2];
  if (b >= 'A' && b <= 'Z') {
    b += 'a' - 'A';
  }
  if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') ||
      (a == 'r' && b == 'e')) {
    return 3;
  }
  return 0;
}

// Return the next piece boundary using the selected pattern and our Latin
// classes: digit grouping, contractions, case, punctuation and whitespace.
static size_t piece_end(
    tokenizer_t* t,
    const unsigned int* cp,
    const unsigned int* cls,
    size_t len,
    size_t start
) {
  size_t end;
  if (t->contractions && !t->case_split) {
    size_t count = contraction_len(cp, len, start);
    if (count) {
      return start + count;
    }
  }
  unsigned int letters = TOKENIZER_LETTER | (t->marks ? TOKENIZER_MARK : 0);
  bool prefix = cp[start] != '\r' && cp[start] != '\n' &&
                !(cls[start] & (TOKENIZER_LETTER | TOKENIZER_NUMBER));
  // Try the optional prefix first; allow it to backtrack for a lone mark.
  for (size_t attempt = 0; attempt <= (size_t)prefix; attempt++) {
    size_t word = start + (prefix && attempt == 0);
    end = word;
    if (!t->case_split) {
      while (end < len && (cls[end] & letters)) {
        end++;
      }
      if (end > word) {
        return end;
      }
    } else {
      // Uppercase/mark* lowercase/mark+, then uppercase/mark+ lowercase*.
      while (end < len && (cls[end] & (TOKENIZER_UPPER | TOKENIZER_OTHER |
                                       TOKENIZER_MARK))) {
        end++;
      }
      size_t middle = end;
      while (end < len && (cls[end] & (TOKENIZER_LOWER | TOKENIZER_OTHER |
                                       TOKENIZER_MARK))) {
        end++;
      }
      bool first = end > middle ||
                   (middle > word &&
                    (cls[middle - 1] & (TOKENIZER_OTHER | TOKENIZER_MARK)));
      if (first || middle > word) {
        if (t->contractions) {
          end += contraction_len(cp, len, end);
        }
        return end;
      }
    }
  }
  if (cls[start] & TOKENIZER_NUMBER) {
    end = start;
    while (end < len && end - start < t->digit_count &&
           (cls[end] & TOKENIZER_NUMBER)) {
      end++;
    }
    return end;
  }
  // Optional ASCII space, punctuation run, then newline (or slash) suffix.
  end = start + (cp[start] == ' ');
  size_t punctuation = end;
  unsigned int excluded = TOKENIZER_SPACE | TOKENIZER_LETTER | TOKENIZER_NUMBER;
  if (t->marks && !t->case_split) {
    excluded |= TOKENIZER_MARK;
  }
  while (end < len && !(cls[end] & excluded)) {
    end++;
  }
  if (end > punctuation) {
    while (end < len && (cp[end] == '\r' || cp[end] == '\n' ||
                         (t->case_split && cp[end] == '/'))) {
      end++;
    }
    return end;
  }
  end = start;
  size_t newline = start;
  while (end < len && (cls[end] & TOKENIZER_SPACE)) {
    if (cp[end] == '\r' || cp[end] == '\n') {
      newline = end + 1;
    }
    end++;
  }
  if (newline > start) {
    return newline;
  }
  // Leave the last space for the following word unless at end of input.
  if (end < len && end - start > 1) {
    return end - 1;
  }
  if (end > start) {
    return end;
  }
  return start + 1;
}

// Merge an already initialized sequence, preserving leftmost rank ties.
static void merge_tokens(
    tokenizer_t* t, size_t start, size_t* token_count, int* token
) {
  while (*token_count > start + 1) {
    size_t best_rank = SIZE_MAX;
    size_t best_index = 0;
    int best_id = -1;
    for (size_t i = start; i + 1 < *token_count; i++) {
      tokenizer_merge_t key = {.left = token[i], .right = token[i + 1]};
      tokenizer_merge_t* merge =
          bsearch(&key, t->merge, t->merge_count, sizeof(key), compare_merges);
      if (merge && merge->rank < best_rank) {
        best_rank = merge->rank;
        best_index = i;
        best_id = merge->result;
      }
    }
    if (best_id < 0) {
      break;
    }
    token[best_index] = best_id;
    for (size_t i = best_index + 1; i + 1 < *token_count; i++) {
      token[i] = token[i + 1];
    }
    (*token_count)--;
  }
}

// Merge only within this piece. ignore_merges allows a direct vocabulary
// lookup of the complete piece before applying ordered pair merges.
static void encode_piece(
    tokenizer_t* t, char* text, size_t len, size_t* token_count, int* token
) {
  if (t->ignore_merges) {
    int id = str_lookup(t, text, len);
    if (id >= 0) {
      token[(*token_count)++] = id;
      return;
    }
  }
  size_t start = *token_count;
  for (size_t i = 0; i < len; i++) {
    token[(*token_count)++] = t->byte_token[(unsigned char)text[i]];
  }
  merge_tokens(t, start, token_count, token);
}

// Metaspace works on Unicode characters, falling back to bytes only when a
// character is absent. TinyLlama prepends a space to every text segment.
static void encode_metaspace(
    tokenizer_t* t, const char* text, size_t len, bool first,
    size_t* token_count, int* token
) {
  if (!len) {
    return;
  }
  size_t start = *token_count;
  int space = str_lookup(t, "\xe2\x96\x81", 3);
  if (t->metaspace.normalizer_type_count ||
      (first && text[0] != ' ' && strncmp(text, "\xe2\x96\x81", 3))) {
    token[(*token_count)++] = space;
  }
  const char* end = text + len;
  while (text < end) {
    const char* next = text;
    unsigned int cp = utf8_read(&next);
    size_t size = (size_t)(next - text);
    int id = cp == ' ' ? space : str_lookup(t, (char*)text, size);
    if (id >= 0) {
      token[(*token_count)++] = id;
    } else {
      for (size_t i = 0; i < size; i++) {
        token[(*token_count)++] = t->byte_token[(unsigned char)text[i]];
      }
    }
    text = next;
  }
  merge_tokens(t, start, token_count, token);
}

static void encode_text(
    tokenizer_t* t,
    const char* text,
    size_t len,
    size_t* token_count,
    int* token
) {
  if (!len) {
    return;
  }
  // At most one codepoint per input byte. Composition only shortens input.
  char* bytes = malloc(len + 1);
  unsigned int* cp = malloc(len * sizeof(*cp));
  unsigned int* cls = malloc(len * sizeof(*cls));
  size_t* offset = malloc((len + 1) * sizeof(*offset));
  if (!bytes || !cp || !cls || !offset) {
    UTIL_DIE("failed to malloc for pre-tokenization");
  }
  memcpy(bytes, text, len);
  bytes[len] = '\0';
  const char* in = bytes;
  size_t count = 0;
  size_t out = 0;
  while (*in) {
    const char* previous = in;
    unsigned int value = utf8_read(&in);
    unsigned int composed =
        count && t->normalize ? compose_latin(cp[count - 1], value) : 0;
    if (composed) {
      cp[count - 1] = composed;
      out = offset[count - 1];
      // All entries in the Latin composition table use two UTF-8 bytes.
      bytes[out++] = 0xc0 | (composed >> 6);
      bytes[out++] = 0x80 | (composed & 63);
    } else {
      offset[count] = out;
      cp[count++] = value;
      size_t size = (size_t)(in - previous);
      memmove(bytes + out, previous, size);
      out += size;
    }
  }
  offset[count] = out;
  for (size_t i = 0; i < count; i++) {
    cls[i] = character_class(cp[i]);
  }
  for (size_t i = 0; i < count;) {
    size_t end = piece_end(t, cp, cls, count, i);
    encode_piece(
        t, bytes + offset[i], offset[end] - offset[i], token_count, token
    );
    i = end;
  }
  free(bytes);
  free(cp);
  free(cls);
  free(offset);
}

// Input must be valid UTF-8 without embedded NUL. Match literal added tokens
// before normalization, choosing the longest match at the earliest position.
// Their single_word, lstrip, rstrip and normalized options must be false.
// Chat formatting is handled by the application.
//
// Tokenizer corrections can change prompt IDs and generated text without
// changing inference arithmetic. Compare builds using the same tokenizer
// behavior or explicit pre-tokenized input.
void tokenizer_tokenize(
    tokenizer_t* t,
    char* text,
    bool bos,
    bool eos,
    size_t* token_count,
    int** token_ptr
) {
  if (!text) {
    UTIL_DIE("cannot encode NULL text");
  }
  size_t len = strlen(text);
  // TinyLlama may add a prefix after every added token. Reserve enough for
  // alternating single-byte added tokens and single-byte text segments.
  size_t token_per_byte_count = t->metaspace.normalizer_type_count ? 2 : 1;
  if (len > (SIZE_MAX / sizeof(**token_ptr) - 3) / token_per_byte_count) {
    UTIL_ERROR("tokenizer input is too large");
  }
  int* token = malloc((len * token_per_byte_count + 3) * sizeof(*token));
  if (!token) {
    UTIL_DIE("failed to malloc for tokens");
  }
  *token_ptr = token;
  *token_count = 0;
  if (bos && t->bos_token_id >= 0) {
    token[(*token_count)++] = t->bos_token_id;
  }
  const char* begin = text;
  const char* end = text + len;
  while (text < end) {
    // Added tokens are literal, non-normalized strings in supported files.
    const char* next = end;
    int added_id = -1;
    size_t added_len = 0;
    for (size_t i = 0; i < t->added_count; i++) {
      int id = t->added_token[i];
      const char* match = strstr(text, t->token_string[id]);
      size_t size = t->token_string_len[id];
      if (match && (match < next || (match == next && size > added_len))) {
        next = match;
        added_id = id;
        added_len = size;
      }
    }
    if (t->metaspace.enabled) {
      encode_metaspace(
          t, text, (size_t)(next - text), text == begin, token_count, token
      );
    } else {
      encode_text(t, text, (size_t)(next - text), token_count, token);
    }
    if (added_id >= 0) {
      token[(*token_count)++] = added_id;
    }
    text = (char*)next + added_len;
  }
  if (eos && t->eos_token_id >= 0) {
    token[(*token_count)++] = t->eos_token_id;
  }
}
