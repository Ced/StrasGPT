%{
  #include "util.h"
  #include "y.tab.h"
  #include <limits.h>
  #include <math.h>
  #include <stdbool.h>
  #include <stdlib.h>
  #include <string.h>
  size_t json_scanner_line_count = 1;
  void dump(char*, char*);
  static bool tokenizer_mode = false;
  static char* decode_json_string(const char* string);
%}

%option prefix="json_scanner_"
%option noyywrap noinput nounput

/* Special mode that turns off returning keyword tokens for all the
 * particular cases of strings (e.g., in default INITIAL mode, the scanner
 * would return MODEL when reading "model", but in KW_AS_STRING mode it
 * would return STRING with value "model"). Functions are provided for
 * mode switching.
 */
%x KW_AS_STRING

integer           [+-]?[0-9]+
float             [+-]?(([0-9]*\.[0-9]+)|([0-9]+\.)|[0-9]+)([eE][+-]?[0-9]+)?

metadata              \"__metadata__\"|\"metadata\"
text_config           \"text_config\"
type                  \"dtype\"
shape                 \"shape\"
offset                \"data_offsets\"
weight_map            \"weight_map\"
bos_token_id          \"bos_token_id\"
eos_token_id          \"eos_token_id\"
embedding_dim         \"hidden_size\"
head_dim              \"head_dim\"
hidden_dim            \"intermediate_size\"
layer_count           \"num_hidden_layers\"
layer_types           \"layer_types\"
full_attention        \"full_attention\"
sliding_attention     \"sliding_attention\"
linear_attention      \"linear_attention\"
model_type            \"model_type\"
q_head_count          \"num_attention_heads\"
kv_head_count         \"num_key_value_heads\"
la_kernel_size        \"linear_conv_kernel_dim\"
la_k_head_dim         \"linear_key_head_dim\"
la_k_head_count       \"linear_num_key_heads\"
la_v_head_dim         \"linear_value_head_dim\"
la_v_head_count       \"linear_num_value_heads\"
mha_output_gate       \"attn_output_gate\"
vocabulary_len        \"vocab_size\"
expert_count          \"num_local_experts\"
expert_per_token_count \"num_experts_per_tok\"
ffn_swiglu_limit      \"swiglu_limit\"
swa_len               \"sliding_window\"
context_len           \"max_position_embeddings\"
epsilon               \"rms_norm_eps\"
rope_theta            \"rope_theta\"
rope_type             \"rope_type\"
rope_factor           \"factor\"
rope_context_len      \"original_max_position_embeddings\"
rope_beta_fast        \"beta_fast\"
rope_beta_slow        \"beta_slow\"
rope_yarn_truncate    \"truncate\"
rope_scaling          \"rope_scaling\"|\"rope_parameters\"
mrope_interleaved     \"mrope_interleaved\"
mrope_section         \"mrope_section\"
partial_rotary_factor \"partial_rotary_factor\"
model                 \"model\"
vocab                 \"vocab\"

string            \"(\\.|[^\"\\])*\"
boolean           true|false
null              null

%%

<INITIAL,KW_AS_STRING>{integer} {
  dump("number (int)", yytext);
  long long val = atoll(yytext);
  yylval.number.is_int = true;
  yylval.number.ival = val;
  yylval.number.fval = val;
  return NUMBER;
}

<INITIAL,KW_AS_STRING>{float} {
  dump("number (float)", yytext);
  double val = atof(yytext);
  double integer_part;
  bool is_int = (modf(val, &integer_part) == 0.0);
  if (is_int && val >= (double)LLONG_MIN && val <= (double)LLONG_MAX) {
    yylval.number.is_int = true;
    yylval.number.ival = val;
  } else {
    yylval.number.is_int = false;
  }
  yylval.number.fval = val;
  return NUMBER;
}

<INITIAL>{type} {
  dump("type", yytext);
  return TYPE;
}

<INITIAL>{shape} {
  dump("shape", yytext);
  return SHAPE;
}

<INITIAL>{offset} {
  dump("offset", yytext);
  return OFFSET;
}

<INITIAL>{metadata} {
  dump("metadata", yytext);
  return METADATA;
}

<INITIAL>{text_config} {
  dump("text_config", yytext);
  return TEXT_CONFIG;
}

<INITIAL>{weight_map} {
  dump("weight_map", yytext);
  return WEIGHT_MAP;
}

<INITIAL>{bos_token_id} {
  dump("bos_token_id", yytext);
  return BOS_TOKEN_ID;
}

<INITIAL>{eos_token_id} {
  dump("eos_token_id", yytext);
  return EOS_TOKEN_ID;
}

<INITIAL>{embedding_dim} {
  dump("embedding_dim", yytext);
  return EMBEDDING_DIM;
}

<INITIAL>{head_dim} {
  dump("head_dim", yytext);
  return HEAD_DIM;
}

<INITIAL>{hidden_dim} {
  dump("hidden_dim", yytext);
  return HIDDEN_DIM;
}

<INITIAL>{layer_count} {
  dump("layer_count", yytext);
  return LAYER_COUNT;
}

<INITIAL>{layer_types} {
  dump("layer_types", yytext);
  return LAYER_TYPES;
}

<INITIAL>{full_attention} {
  dump("full_attention", yytext);
  return FULL_ATTENTION;
}

<INITIAL>{sliding_attention} {
  dump("sliding_attention", yytext);
  return SLIDING_ATTENTION;
}

<INITIAL>{linear_attention} {
  dump("linear_attention", yytext);
  return LINEAR_ATTENTION;
}

<INITIAL>{model_type} {
  dump("model_type", yytext);
  return MODEL_TYPE;
}

<INITIAL>{q_head_count} {
  dump("q_head_count", yytext);
  return Q_HEAD_COUNT;
}

<INITIAL>{kv_head_count} {
  dump("kv_head_count", yytext);
  return KV_HEAD_COUNT;
}

<INITIAL>{la_kernel_size} {
  dump("la_kernel_size", yytext);
  return LA_KERNEL_SIZE;
}

<INITIAL>{la_k_head_dim} {
  dump("la_k_head_dim", yytext);
  return LA_K_HEAD_DIM;
}

<INITIAL>{la_k_head_count} {
  dump("la_k_head_count", yytext);
  return LA_K_HEAD_COUNT;
}

<INITIAL>{la_v_head_dim} {
  dump("la_v_head_dim", yytext);
  return LA_V_HEAD_DIM;
}

<INITIAL>{la_v_head_count} {
  dump("la_v_head_count", yytext);
  return LA_V_HEAD_COUNT;
}

<INITIAL>{mha_output_gate} {
  dump("mha_output_gate", yytext);
  return MHA_OUTPUT_GATE;
}

<INITIAL>{vocabulary_len} {
  dump("vocabulary_len", yytext);
  return VOCABULARY_LEN;
}

<INITIAL>{expert_count} {
  dump("expert_count", yytext);
  return EXPERT_COUNT;
}

<INITIAL>{expert_per_token_count} {
  dump("expert_per_token_count", yytext);
  return EXPERT_PER_TOKEN_COUNT;
}

<INITIAL>{ffn_swiglu_limit} {
  dump("ffn_swiglu_limit", yytext);
  return FFN_SWIGLU_LIMIT;
}

<INITIAL>{swa_len} {
  dump("swa_len", yytext);
  return SWA_LEN;
}

<INITIAL>{context_len} {
  dump("context_len", yytext);
  return CONTEXT_LEN;
}

<INITIAL>{epsilon} {
  dump("epsilon", yytext);
  return EPSILON;
}

<INITIAL>{rope_theta} {
  dump("rope_theta", yytext);
  return ROPE_THETA;
}

<INITIAL>{rope_type} {
  dump("rope_type", yytext);
  return ROPE_TYPE;
}

<INITIAL>{rope_factor} {
  dump("rope_factor", yytext);
  return ROPE_FACTOR;
}

<INITIAL>{rope_context_len} {
  dump("rope_context_len", yytext);
  return ROPE_CONTEXT_LEN;
}

<INITIAL>{rope_beta_fast} {
  dump("rope_beta_fast", yytext);
  return ROPE_BETA_FAST;
}

<INITIAL>{rope_beta_slow} {
  dump("rope_beta_slow", yytext);
  return ROPE_BETA_SLOW;
}

<INITIAL>{rope_yarn_truncate} {
  dump("rope_yarn_truncate", yytext);
  return ROPE_YARN_TRUNCATE;
}

<INITIAL>{rope_scaling} {
  dump("rope_scaling", yytext);
  return ROPE_SCALING;
}

<INITIAL>{mrope_interleaved} {
  dump("mrope_interleaved", yytext);
  return MROPE_INTERLEAVED;
}

<INITIAL>{mrope_section} {
  dump("mrope_section", yytext);
  return MROPE_SECTION;
}

<INITIAL>{partial_rotary_factor} {
  dump("partial_rotary_factor", yytext);
  return PARTIAL_ROTARY_FACTOR;
}

<INITIAL>{model} {
  dump("model", yytext);
  return MODEL;
}

<INITIAL>{vocab} {
  dump("vocab", yytext);
  return VOCAB;
}

<INITIAL,KW_AS_STRING>{string} {
  dump("string", yytext);
  if (tokenizer_mode && YY_START == INITIAL) {
    if (!strcmp(yytext, "\"type\"")) return BPE_TYPE;
    if (!strcmp(yytext, "\"merges\"")) return MERGES;
    if (!strcmp(yytext, "\"added_tokens\"")) return ADDED_TOKENS;
    if (!strcmp(yytext, "\"pre_tokenizer\"")) return PRE_TOKENIZER;
    if (!strcmp(yytext, "\"normalizer\"")) return NORMALIZER;
    if (!strcmp(yytext, "\"ignore_merges\"")) return IGNORE_MERGES;
  }
  yylval.string = decode_json_string(yytext);
  return STRING;
}

<INITIAL,KW_AS_STRING>{boolean} {
  dump("boolean", yytext);
  yylval.boolean = (strcmp(yytext, "true") == 0) ? true : false;
  return BOOLEAN;
}

<INITIAL,KW_AS_STRING>{null} {
  dump("null", yytext);
  return NULL_;
}

<INITIAL,KW_AS_STRING>[{}\[\],:] {
  dump("symbol", yytext);
  return yytext[0];
}

<INITIAL,KW_AS_STRING>[ \t\r] { }

<INITIAL,KW_AS_STRING>[\n] {
  json_scanner_line_count++;
}

<INITIAL,KW_AS_STRING>. { UTIL_ERROR("unexpected character"); }

%%

void dump(char* token, char* val) {
  (void)token;
  (void)val;
  //fprintf(stderr, "Lex: %s (%s)\n", token, val);
}

void json_scanner_reset(void) {
  json_scanner_line_count = 1;
  tokenizer_mode = false;
  BEGIN(INITIAL);
}

void json_scanner_enter_kw_as_string_mode(void) {
  BEGIN(KW_AS_STRING);
}

void json_scanner_leave_kw_as_string_mode(void) {
  BEGIN(INITIAL);
}

void json_scanner_enter_tokenizer_mode(void) {
  tokenizer_mode = true;
}

static unsigned int json_hex(const char* string) {
  unsigned int value = 0;
  for (size_t i = 0; i < 4; i++) {
    unsigned char c = string[i];
    unsigned int digit;
    if (c >= '0' && c <= '9') {
      digit = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      digit = c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
      digit = c - 'A' + 10;
    } else {
      UTIL_ERROR("invalid JSON Unicode escape");
    }
    value = 16 * value + digit;
  }
  return value;
}

// Unescape before interpreting ByteLevel's printable Unicode alphabet.
static char* decode_json_string(const char* string) {
  size_t len = strlen(string) - 1; // Position of closing quote
  char* result = malloc(len + 1);
  if (!result) {
    UTIL_DIE("failed to malloc for JSON string");
  }
  size_t out = 0;
  for (size_t i = 1; i < len; i++) {
    unsigned char c = string[i];
    if (c < 0x20) {
      UTIL_ERROR("control character in JSON string");
    }
    if (c != '\\') {
      result[out++] = c;
      continue;
    }
    c = string[++i];
    switch (c) {
      case '"':
      case '\\':
      case '/':
        result[out++] = c;
        break;
      case 'b':
        result[out++] = '\b';
        break;
      case 'f':
        result[out++] = '\f';
        break;
      case 'n':
        result[out++] = '\n';
        break;
      case 'r':
        result[out++] = '\r';
        break;
      case 't':
        result[out++] = '\t';
        break;
      case 'u': {
        if (i + 4 >= len) {
          UTIL_ERROR("short JSON Unicode escape");
        }
        unsigned int cp = json_hex(string + i + 1);
        i += 4;
        if (cp >= 0xd800 && cp <= 0xdbff) {
          if (i + 6 >= len || string[i + 1] != '\\' || string[i + 2] != 'u') {
            UTIL_ERROR("missing JSON low surrogate");
          }
          unsigned int low = json_hex(string + i + 3);
          if (low < 0xdc00 || low > 0xdfff) {
            UTIL_ERROR("invalid JSON low surrogate");
          }
          cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
          i += 6;
        } else if (cp >= 0xdc00 && cp <= 0xdfff) {
          UTIL_ERROR("unexpected JSON low surrogate");
        }
        // The parser uses C strings. ByteLevel encodes NUL as U+0100.
        if (cp == 0) {
          UTIL_ERROR("NUL in JSON string is unsupported");
        }
        if (cp < 0x80) {
          result[out++] = cp;
        } else {
          if (cp >= 0x10000) {
            result[out++] = 0xf0 | (cp >> 18);
            result[out++] = 0x80 | ((cp >> 12) & 63);
          } else if (cp >= 0x800) {
            result[out++] = 0xe0 | (cp >> 12);
          } else {
            result[out++] = 0xc0 | (cp >> 6);
          }
          if (cp >= 0x800) {
            result[out++] = 0x80 | ((cp >> 6) & 63);
          }
          result[out++] = 0x80 | (cp & 63);
        }
        break;
      }
      default:
        UTIL_ERROR("invalid JSON escape");
    }
  }
  result[out] = '\0';
  return result;
}
