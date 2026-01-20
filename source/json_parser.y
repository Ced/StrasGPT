%{
  #include "safetensors.h"
  #include "tokenizer.h"
  #include "util.h"
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  void yyerror(char*);
  int yylex(void);
  int json_scanner_lex(void);
  int json_scanner_restart(FILE*);
  void json_scanner_reset(void);
  void json_scanner_enter_kw_as_string_mode(void);
  void json_scanner_leave_kw_as_string_mode(void);
  safetensors_t* parser_parse_safetensors(const char*);
  tokenizer_t* parser_parse_tokenizer(const char*);
  extern FILE* json_scanner_in;
  extern size_t json_scanner_line_count;

  // Parser state
  enum {
    PARSER_MODE_CONFIG = 0,
    PARSER_MODE_INDEX,
    PARSER_MODE_SAFETENSORS,
    PARSER_MODE_TOKENIZER,
    PARSER_MODE_STARTED
  } parser_mode = PARSER_MODE_CONFIG;
  char parser_path[SAFETENSORS_MAX_STRING];
  safetensors_t* parser_safetensors;
  tokenizer_t* parser_tokenizer;
  size_t parser_tensor = 0;
  size_t parser_dim = 0;
  size_t parser_file = 0;
  size_t parser_header_len = 0;
%}

%union {
  char* string;     // For STRING
  struct {          // For NUMBER
    char is_int;    // - true if this lexeme is/equals an integer
    long long ival; // - integer value, valid iff is_int == true
    double fval;    // - floating point value, always filled
  } number;
  char boolean;     // For BOOLEAN
}

%token <string> STRING
%token <number> NUMBER
%token <boolean> BOOLEAN
%token NULL_ TYPE SHAPE OFFSET METADATA TEXT_CONFIG WEIGHT_MAP
%token BOS_TOKEN_ID EOS_TOKEN_ID
%token EMBEDDING_DIM HEAD_DIM HIDDEN_DIM LAYER_COUNT MODEL_TYPE Q_HEAD_COUNT
%token KV_HEAD_COUNT VOCABULARY_LEN CONTEXT_LEN MODEL VOCAB
%token ROPE_THETA ROPE_SCALING MROPE_INTERLEAVED MROPE_SECTION
%token MODE_CONFIG MODE_INDEX MODE_SAFETENSORS MODE_TOKENIZER
%start entry

%%

entry
  : MODE_CONFIG config
  | MODE_INDEX index
  | MODE_SAFETENSORS safetensors
  | MODE_TOKENIZER tokenizer
  ;

// +--------------------------------------------------------------------------+
// |                           config.json grammar                            |
// +--------------------------------------------------------------------------+

config
  : '{' config_member_list '}'
  | '{' '}'
  ;

config_member_list
  : config_member_list ',' config_member
  | config_member
  ;

config_member
  : BOS_TOKEN_ID ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer BOS token id");
        YYABORT;
      }
      parser_safetensors->bos_token_id = (int)$3.ival;
    }
  | EOS_TOKEN_ID ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer EOS token id");
        YYABORT;
      }
      parser_safetensors->eos_token_id = (int)$3.ival;
    }
  | EMBEDDING_DIM ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer embedding_dim value");
        YYABORT;
      }
      parser_safetensors->embedding_dim = $3.ival;
    }
  | HEAD_DIM ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer head_dim value");
        YYABORT;
      }
      parser_safetensors->head_dim = $3.ival;
    }
  | HIDDEN_DIM ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer hidden_dim value");
        YYABORT;
      }
      parser_safetensors->hidden_dim = $3.ival;
    }
  | LAYER_COUNT ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer layer_count value");
        YYABORT;
      }
      parser_safetensors->layer_count = $3.ival;
    }
  | MODEL_TYPE ':' STRING
    {
      free($3);
    }
  | Q_HEAD_COUNT ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer q_head_count value");
        YYABORT;
      }
      parser_safetensors->q_head_count = $3.ival;
    }
  | KV_HEAD_COUNT ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer kv_head_count value");
        YYABORT;
      }
      parser_safetensors->kv_head_count = $3.ival;
    }
  | VOCABULARY_LEN ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer vocabulary_len value");
        YYABORT;
      }
      parser_safetensors->vocabulary_len = $3.ival;
    }
  | CONTEXT_LEN ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer context_len value");
        YYABORT;
      }
      parser_safetensors->context_len = $3.ival;
    }
  | ROPE_THETA ':' NUMBER
    {
      parser_safetensors->rope_theta = $3.fval;
    }
  | ROPE_SCALING ':' '{' rope_scaling_member_list '}'
  | ROPE_SCALING ':' NULL_
  | TYPE ':' STRING
    {
      free($3);
    }
  | TEXT_CONFIG ':' config
  | STRING ':'
    {
      // We enter special mode where keyword strings (e.g., "model" where
      // Lex would return MODEL token) are considered as normal strings,
      // to avoid issues if they are part of the model vocabulary
      json_scanner_enter_kw_as_string_mode();
    }
    json_value
    {
      free($1);
      json_scanner_leave_kw_as_string_mode();
    }
  ;

rope_scaling_member_list
  : rope_scaling_member_list ',' rope_scaling_member
  | rope_scaling_member
  ;

rope_scaling_member
  : MROPE_SECTION ':' '[' mrope_section_list ']'
  | MROPE_INTERLEAVED ':' BOOLEAN
    {
      parser_safetensors->rope_interleaved = $3;
    }
  | STRING ':' json_value
    {
      free($1);
    }

mrope_section_list
  : mrope_section_list ',' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer mrope section value");
        YYABORT;
      }
      if (parser_safetensors->mrope_section_count >=
          SAFETENSORS_MAX_MROPE_SECTION_COUNT) {
        yyerror("too many mrope sections");
        YYABORT;
      }
      parser_safetensors
          ->mrope_section[parser_safetensors->mrope_section_count] = $3.ival;
      parser_safetensors->mrope_section_count++;

    }
  | NUMBER
    {
      if (!$1.is_int) {
        yyerror("non integer mrope section value");
        YYABORT;
      }
      if (parser_safetensors->mrope_section_count >=
          SAFETENSORS_MAX_MROPE_SECTION_COUNT) {
        yyerror("too many mrope sections");
        YYABORT;
      }
      parser_safetensors
          ->mrope_section[parser_safetensors->mrope_section_count] = $1.ival;
      parser_safetensors->mrope_section_count++;
    }
  ;

// +--------------------------------------------------------------------------+
// |                  model.safetensors.index.json grammar                    |
// +--------------------------------------------------------------------------+

index
  : '{' index_member_list '}'
  | '{' '}'
  ;

index_member_list
  : index_member_list ',' index_member
  | index_member
  ;

index_member
  : METADATA ':' json_value
  | WEIGHT_MAP ':' '{' index_weight_map_list '}'
  ;

index_weight_map_list
  : index_weight_map_list ',' index_weight_map
  | index_weight_map
  ;

index_weight_map
  : STRING ':' STRING
    {
      safetensors_file_lookup(parser_safetensors, parser_path, $3);
      free($1);
      free($3);
    }
  ;

// +--------------------------------------------------------------------------+
// |                            safetensors grammar                           |
// +--------------------------------------------------------------------------+

safetensors
  : '{' safetensors_member_list '}' { YYACCEPT; }
  | '{' '}'                         { YYACCEPT; }
  ;

safetensors_member_list
  : safetensors_member_list ',' safetensors_member
  | safetensors_member
  ;

safetensors_member
  : METADATA ':' json_value
  | STRING ':'
    {
      if (parser_tensor >= SAFETENSORS_MAX_TENSOR_COUNT) {
        yyerror("too many tensors");
        YYABORT;
      }
    }
    '{' safetensors_property_list '}'
    {
      parser_safetensors->tensor[parser_tensor].name = $1;
      parser_safetensors->tensor_count++;
      parser_tensor++;
    }
  ;

safetensors_property_list
  : safetensors_property_list ',' safetensors_property
  | safetensors_property
  ;

safetensors_property
  : TYPE ':' STRING
    {
      parser_safetensors->tensor[parser_tensor].type =
          safetensors_type_from_string($3);
      free($3);
    }
  | SHAPE ':' safetensors_shape
    {
      parser_safetensors->tensor[parser_tensor].dim_count = parser_dim;
      parser_dim = 0;
    }
  | OFFSET ':' '[' NUMBER ',' NUMBER ']'
    {
      if (!$4.is_int || !$6.is_int) {
        yyerror("non integer offset value");
        YYABORT;
      }
      size_t size = $6.ival - $4.ival;
      size_t offset = 8 + parser_header_len + $4.ival; // +8 for header length
      parser_safetensors->tensor[parser_tensor].offset = offset;
      parser_safetensors->tensor[parser_tensor].size = size;
      parser_safetensors->tensor[parser_tensor].file = parser_file;
    }
  ;

safetensors_shape
  : '[' safetensors_dimension_list ']'
  | '[' ']'
  ;

safetensors_dimension_list
  : safetensors_dimension_list ',' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer dimension value");
        YYABORT;
      }
      if (parser_dim >= SAFETENSORS_MAX_DIM_COUNT) {
        yyerror("too many tensor dimensions");
        YYABORT;
      }
      parser_safetensors->tensor[parser_tensor].dim[parser_dim] = $3.ival;
      parser_dim++;

    }
  | NUMBER
    {
      if (!$1.is_int) {
        yyerror("non integer dimension value");
        YYABORT;
      }
      if (parser_dim >= SAFETENSORS_MAX_DIM_COUNT) {
        yyerror("too many tensor dimensions");
        YYABORT;
      }
      parser_safetensors->tensor[parser_tensor].dim[parser_dim] = $1.ival;
      parser_dim++;
    }
  ;

// +--------------------------------------------------------------------------+
// |                             tokenizer grammar                            |
// +--------------------------------------------------------------------------+

tokenizer
  : '{' tokenizer_member_list '}'
  | '{' '}'
  ;

tokenizer_member_list
  : tokenizer_member_list ',' tokenizer_member
  | tokenizer_member
  ;

tokenizer_member
  : MODEL ':' '{' tokenizer_model_member_list '}'
  | STRING ':' json_value
    {
      free($1);
    }
  ;

tokenizer_model_member_list
  : tokenizer_model_member_list ',' tokenizer_model_member
  | tokenizer_model_member
  ;

tokenizer_model_member
  : VOCAB ':' '{'
    {
      // We enter special mode where keyword strings (e.g., "model" where
      // Lex would return MODEL token) are considered as normal strings,
      // to avoid issues if they are part of the model vocabulary
      json_scanner_enter_kw_as_string_mode();
    }
    tokenizer_vocab_member_list
    {
      // Back to normal mode
      json_scanner_leave_kw_as_string_mode();
    }
    '}'
  | STRING ':'
    {
      // See above
      json_scanner_enter_kw_as_string_mode();
    }
    json_value
    {
      free($1);
      json_scanner_leave_kw_as_string_mode();
    }
  ;

tokenizer_vocab_member_list
  : tokenizer_vocab_member_list ',' tokenizer_vocab_member
  | tokenizer_vocab_member
  ;

tokenizer_vocab_member
  : STRING ':' NUMBER
    {
      size_t i = parser_tokenizer->token_string_count;
      if (i >= TOKENIZER_MAX_TOKEN_STRING) {
        yyerror("too many token strings");
        YYABORT;
      }
      parser_tokenizer->token_string[i] = $1;
      parser_tokenizer->score[i] = (float)$3.fval;
      parser_tokenizer->token_string_count++;
    }
  ;

// +--------------------------------------------------------------------------+
// |                            general json grammar                          |
// +--------------------------------------------------------------------------+

json
  : '{' json_member_list '}'
  | '{' '}'
  ;

json_list
  : '[' json_value_list ']'
  | '[' ']'
  ;

json_value_list
  : json_value_list ',' json_value
  | json_value
  ;

json_member_list
  : json_member_list ',' json_member
  | json_member
  ;

json_member
  : STRING ':' json_value { free($1); }
  ;

json_value
  : STRING { free($1); }
  | NUMBER
  | BOOLEAN
  | NULL_
  | json
  | json_list
  ;

%%

// Error handling function
void yyerror(char* err) {
  char msg[SAFETENSORS_MAX_STRING];
  snprintf(
    msg, SAFETENSORS_MAX_STRING, "line %zu: %s", json_scanner_line_count, err
  );
  UTIL_ERROR(msg);
}

// If parsing is not started yet, return the appropriate mode token,
// otherwise call the scanner as usual.
// Note: this is a wrapper around json_scanner_lex to handle an initial
// mode token that allows to use a single grammar for several file
// formats. Here this makes sense as the formats are flavors of JSON.
// This allows to share the scanner between multiple modes and
// switching mode at the start of parsing.
int yylex(void) {
  switch (parser_mode) {
    case PARSER_MODE_CONFIG:
      parser_mode = PARSER_MODE_STARTED;
      return MODE_CONFIG;
    case PARSER_MODE_INDEX:
      parser_mode = PARSER_MODE_STARTED;
      return MODE_INDEX;
    case PARSER_MODE_SAFETENSORS:
      parser_mode = PARSER_MODE_STARTED;
      return MODE_SAFETENSORS;
    case PARSER_MODE_TOKENIZER:
      parser_mode = PARSER_MODE_STARTED;
      return MODE_TOKENIZER;
    case PARSER_MODE_STARTED:
      return json_scanner_lex();
  }
  // Should not happen
  return json_scanner_lex();
}

// Parse the safetensors files in the given path
// and return the corresponding safetensors_t structure.
safetensors_t* parser_parse_safetensors(const char* path) {
  char fullpath[SAFETENSORS_MAX_STRING];
  parser_safetensors = safetensors_malloc();

  // Let's parse the config file first
  #ifdef DEBUG
  fprintf(stderr, "[StrasGPT] Parsing config file... ");
  #endif
  snprintf(
    fullpath, sizeof(fullpath), "%s/%s", path, SAFETENSORS_FILE_CONFIG
  );
  json_scanner_in = fopen(fullpath, "rb");
  if (!json_scanner_in) {
    fprintf(stderr, "[StrasGPT] Error: failed to open file %s\n", fullpath);
    fprintf(stderr, "Use \"-m <path_to_model_directory>\" option\n");
    exit(EXIT_FAILURE);
  }
  parser_mode = PARSER_MODE_CONFIG;
  json_scanner_reset();
  json_scanner_restart(json_scanner_in);
  yyparse();
  fclose(json_scanner_in);
  #ifdef DEBUG
  fprintf(stderr, "Done\n");
  #endif

  // Then let's parse the index file, if any
  #ifdef DEBUG
  fprintf(stderr, "[StrasGPT] Parsing index file... ");
  #endif
  snprintf(
    fullpath, sizeof(fullpath), "%s/%s", path, SAFETENSORS_FILE_INDEX
  );
  json_scanner_in = fopen(fullpath, "rb");
  if (!json_scanner_in) {
    // If not present, we expect a single safetensors file
    parser_safetensors->file_count = 1;
    snprintf(
      fullpath, sizeof(fullpath), "%s/%s", path, SAFETENSORS_FILE_SAFETENSORS
    );
    parser_safetensors->file[0] = strdup(fullpath);
  } else {
    parser_mode = PARSER_MODE_INDEX;
    snprintf(parser_path, sizeof(parser_path), "%s", path);
    json_scanner_reset();
    json_scanner_restart(json_scanner_in);
    yyparse();
    fclose(json_scanner_in);
  }
  #ifdef DEBUG
  fprintf(stderr, "Done\n");
  #endif

  // Finally let's parse the safetensors file(s)
  for (size_t i = 0; i < parser_safetensors->file_count; i++) {
    parser_file = i;
    #ifdef DEBUG
    fprintf(
      stderr,
      "[StrasGPT] Parsing safetensors file %s... ",
      parser_safetensors->file[i]
    );
    #endif
    snprintf(fullpath, sizeof(fullpath), "%s", parser_safetensors->file[i]);
    json_scanner_in = fopen(fullpath, "rb");
    if (!json_scanner_in) {
      UTIL_DIE("failed to open safetensors file");
    }

    // First read header's length
    uint64_t header_len;
    if (fread(&header_len, 1, 8, json_scanner_in) != 8) {
      fclose(json_scanner_in);
      UTIL_DIE("failed to read safetensors header length");
    }

    // Then read JSON header
    parser_mode = PARSER_MODE_SAFETENSORS;
    parser_header_len = (size_t)header_len;
    json_scanner_reset();
    json_scanner_restart(json_scanner_in);
    yyparse();
    fclose(json_scanner_in);
    #ifdef DEBUG
    fprintf(stderr, "Done\n");
    #endif
  }

  // In some config files, head_dim is not specified. In that case,
  // default value is embedding_dim / q_head_count
  if (parser_safetensors->head_dim == 0) {
    parser_safetensors->head_dim =
        parser_safetensors->embedding_dim / parser_safetensors->q_head_count;
  }

  return parser_safetensors;
}

// Parse the tokenizer file in the given path
// and return the corresponding tokenizer_t structure.
tokenizer_t* parser_parse_tokenizer(const char* path) {
  char fullpath[SAFETENSORS_MAX_STRING];
  parser_tokenizer = tokenizer_malloc();

  #ifdef DEBUG
  fprintf(stderr, "[StrasGPT] Parsing tokenizer file... ");
  #endif
  snprintf(fullpath, sizeof(fullpath), "%s/%s", path, TOKENIZER_FILE);
  json_scanner_in = fopen(fullpath, "rb");
  if (!json_scanner_in) {
    UTIL_DIE("Failed to open tokenizer file");
  }
  parser_mode = PARSER_MODE_TOKENIZER;
  json_scanner_reset();
  json_scanner_restart(json_scanner_in);
  yyparse();
  fclose(json_scanner_in);
  #ifdef DEBUG
  fprintf(stderr, "Done\n");
  #endif

  return parser_tokenizer;
}
