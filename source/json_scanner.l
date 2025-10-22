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

integer        [+-]?[0-9]+
float          [+-]?(([0-9]*\.[0-9]+)|([0-9]+\.)|[0-9]+)([eE][+-]?[0-9]+)?

metadata       \"__metadata__\"|\"metadata\"
text_config    \"text_config\"
type           \"dtype\"
shape          \"shape\"
offset         \"data_offsets\"
weight_map     \"weight_map\"
bos_token_id   \"bos_token_id\"
eos_token_id   \"eos_token_id\"
embedding_dim  \"hidden_size\"
head_dim       \"head_dim\"
hidden_dim     \"intermediate_size\"
layer_count    \"num_hidden_layers\"
model_type     \"model_type\"
q_head_count   \"num_attention_heads\"
kv_head_count  \"num_key_value_heads\"
vocabulary_len \"vocab_size\"
context_len    \"max_position_embeddings\"
rope_theta     \"rope_theta\"
rope_scaling   \"rope_scaling\"
mrope_section  \"mrope_section\"
model          \"model\"
vocab          \"vocab\"

string         \"(\\.|[^\"\\])*\"
boolean        true|false
null           null

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

<INITIAL>{vocabulary_len} {
  dump("vocabulary_len", yytext);
  return VOCABULARY_LEN;
}

<INITIAL>{context_len} {
  dump("context_len", yytext);
  return CONTEXT_LEN;
}

<INITIAL>{rope_theta} {
  dump("rope_theta", yytext);
  return ROPE_THETA;
}

<INITIAL>{rope_scaling} {
  dump("rope_scaling", yytext);
  return ROPE_SCALING;
}

<INITIAL>{mrope_section} {
  dump("mrope_section", yytext);
  return MROPE_SECTION;
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
  yylval.string = strdup(yytext + 1); // skip opening quote
  yylval.string[strlen(yylval.string) - 1] = '\0'; // remove closing quote
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

<INITIAL,KW_AS_STRING>[ \t] { }

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
  json_scanner_line_count = 0;
}

void json_scanner_enter_kw_as_string_mode(void) {
  BEGIN(KW_AS_STRING);
}

void json_scanner_leave_kw_as_string_mode(void) {
  BEGIN(INITIAL);
}
