%{
  #include "safetensors.h"
  #include "tokenizer.h"
  #include "util.h"
  #include <limits.h>
  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  void yyerror(char*);
  int yylex(void);
  int json_scanner_lex(void);
  int json_scanner_restart(FILE*);
  void json_scanner_reset(void);
  void json_scanner_enter_tokenizer_mode(void);
  static void tokenizer_metadata_string(char* key, char* value);
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
  size_t parser_layer_count = 0;
  size_t parser_header_len = 0;
  int parser_metadata_mode;
  int parser_added_id;
  char* parser_added_string;
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
%token SWA_LEN EXPERT_COUNT EXPERT_PER_TOKEN_COUNT FFN_SWIGLU_LIMIT
%token ROPE_TYPE ROPE_FACTOR ROPE_CONTEXT_LEN
%token ROPE_BETA_FAST ROPE_BETA_SLOW ROPE_YARN_TRUNCATE
%token EPSILON ROPE_THETA ROPE_SCALING MROPE_INTERLEAVED MROPE_SECTION PARTIAL_ROTARY_FACTOR
%token LAYER_TYPES LA_KERNEL_SIZE LA_K_HEAD_DIM LA_K_HEAD_COUNT
%token LA_V_HEAD_DIM LA_V_HEAD_COUNT
%token MHA_OUTPUT_GATE
%token FULL_ATTENTION LINEAR_ATTENTION SLIDING_ATTENTION
%token BPE_TYPE MERGES ADDED_TOKENS PRE_TOKENIZER NORMALIZER IGNORE_MERGES
%token BYTE_FALLBACK DECODER
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

eos_token_ids
  : eos_token
  | '[' eos_token_list ']'
  ;

eos_token_list
  : eos_token
  | eos_token_list ',' eos_token
  ;

eos_token
  : NUMBER
    {
      if (!$1.is_int || $1.ival < 0 || $1.ival > INT_MAX) {
        yyerror("invalid EOS token id");
        YYABORT;
      }
      size_t count = parser_safetensors->eos_token_count;
      if (count == SAFETENSORS_MAX_EOS_TOKEN_COUNT) {
        yyerror("too many EOS token ids");
        YYABORT;
      }
      parser_safetensors->eos_token_id[count] = (int)$1.ival;
      parser_safetensors->eos_token_count++;
    }
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
  | EOS_TOKEN_ID ':' { parser_safetensors->eos_token_count = 0; }
    eos_token_ids
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
      if ($3.ival < 0 || $3.ival > SAFETENSORS_MAX_LAYER_COUNT) {
        yyerror("layer_count exceeds supported maximum");
        YYABORT;
      }
      parser_safetensors->layer_count = $3.ival;
    }
  | LAYER_TYPES ':' '[' layer_type_list ']'
  | LAYER_TYPES ':' '[' ']'
  | MODEL_TYPE ':' STRING
    {
      parser_safetensors->model_type = strdup($3);
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
  | LA_KERNEL_SIZE ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid linear_conv_kernel_dim value");
        YYABORT;
      }
      parser_safetensors->la_kernel_size = $3.ival;
    }
  | LA_K_HEAD_DIM ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid linear_key_head_dim value");
        YYABORT;
      }
      parser_safetensors->la_k_head_dim = $3.ival;
    }
  | LA_K_HEAD_COUNT ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid linear_num_key_heads value");
        YYABORT;
      }
      parser_safetensors->la_k_head_count = $3.ival;
    }
  | LA_V_HEAD_DIM ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid linear_value_head_dim value");
        YYABORT;
      }
      parser_safetensors->la_v_head_dim = $3.ival;
    }
  | LA_V_HEAD_COUNT ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid linear_num_value_heads value");
        YYABORT;
      }
      parser_safetensors->la_v_head_count = $3.ival;
    }
  | MHA_OUTPUT_GATE ':' BOOLEAN
    {
      parser_safetensors->mha_output_gate = $3;
    }
  | VOCABULARY_LEN ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer vocabulary_len value");
        YYABORT;
      }
      parser_safetensors->vocabulary_len = $3.ival;
    }
  | EXPERT_COUNT ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid expert_count value");
        YYABORT;
      }
      parser_safetensors->expert_count = $3.ival;
    }
  | EXPERT_PER_TOKEN_COUNT ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid expert_per_token_count value");
        YYABORT;
      }
      parser_safetensors->expert_per_token_count = $3.ival;
    }
  | FFN_SWIGLU_LIMIT ':' NUMBER
    {
      if (!isfinite((float)$3.fval) || $3.fval < 0) {
        yyerror("invalid ffn_swiglu_limit value");
        YYABORT;
      }
      parser_safetensors->ffn_swiglu_limit = $3.fval;
    }
  | SWA_LEN ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid swa_len value");
        YYABORT;
      }
      parser_safetensors->swa_len = $3.ival;
    }
  | SWA_LEN ':' NULL_
    {
      parser_safetensors->swa_len = 0;
    }
  | CONTEXT_LEN ':' NUMBER
    {
      if (!$3.is_int) {
        yyerror("non integer context_len value");
        YYABORT;
      }
      parser_safetensors->context_len = $3.ival;
    }
  | EPSILON ':' NUMBER
    {
      parser_safetensors->epsilon = $3.fval;
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

layer_type_list
  : layer_type_list ',' layer_type
  | layer_type
  ;

layer_type
  : FULL_ATTENTION
    {
      if (parser_layer_count >= SAFETENSORS_MAX_LAYER_COUNT) {
        yyerror("too many layers");
        YYABORT;
      }
      parser_safetensors->layer_type[parser_layer_count] =
          SAFETENSORS_LAYER_TYPE_FA;
      parser_layer_count++;
    }
  | LINEAR_ATTENTION
    {
      if (parser_layer_count >= SAFETENSORS_MAX_LAYER_COUNT) {
        yyerror("too many layers");
        YYABORT;
      }
      parser_safetensors->layer_type[parser_layer_count] =
          SAFETENSORS_LAYER_TYPE_LA;
      parser_layer_count++;
    }
  | SLIDING_ATTENTION
    {
      if (parser_layer_count >= SAFETENSORS_MAX_LAYER_COUNT) {
        yyerror("too many layers");
        YYABORT;
      }
      parser_safetensors->layer_type[parser_layer_count] =
          SAFETENSORS_LAYER_TYPE_SWA;
      parser_layer_count++;
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
  | ROPE_THETA ':' NUMBER
    {
      parser_safetensors->rope_theta = $3.fval;
    }
  | PARTIAL_ROTARY_FACTOR ':' NUMBER
    {
      parser_safetensors->partial_rotary_factor = $3.fval;
    }
  | ROPE_TYPE ':' STRING
    {
      parser_safetensors->rope_yarn = strcmp($3, "yarn") == 0;
      free($3);
    }
  | ROPE_CONTEXT_LEN ':' NUMBER
    {
      if (!$3.is_int || $3.ival < 0) {
        yyerror("invalid rope_context_len value");
        YYABORT;
      }
      parser_safetensors->rope_context_len = $3.ival;
    }
  | ROPE_YARN_TRUNCATE ':' BOOLEAN
    {
      parser_safetensors->rope_yarn_truncate = $3;
    }
  | ROPE_FACTOR ':' NUMBER
    {
      if (!isfinite((float)$3.fval) || $3.fval <= 0) {
        yyerror("invalid rope_factor value");
        YYABORT;
      }
      parser_safetensors->rope_factor = $3.fval;
    }
  | ROPE_BETA_FAST ':' NUMBER
    {
      if (!isfinite((float)$3.fval) || $3.fval <= 0) {
        yyerror("invalid rope_beta_fast value");
        YYABORT;
      }
      parser_safetensors->rope_beta_fast = $3.fval;
    }
  | ROPE_BETA_SLOW ':' NUMBER
    {
      if (!isfinite((float)$3.fval) || $3.fval <= 0) {
        yyerror("invalid rope_beta_slow value");
        YYABORT;
      }
      parser_safetensors->rope_beta_slow = $3.fval;
    }
  | STRING ':' json_value
    {
      free($1);
    }
  ;

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
      parser_safetensors->tensor[parser_tensor].name = $1;
    }
    '{' safetensors_property_list '}'
    {
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
          safetensors_type_from_string(
              $3, parser_safetensors->model_type,
              parser_safetensors->tensor[parser_tensor].name
          );
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
  | ADDED_TOKENS ':' '['
    { json_scanner_enter_kw_as_string_mode(); }
    tokenizer_added_list ']'
    { json_scanner_leave_kw_as_string_mode(); }
  | PRE_TOKENIZER ':'
    {
      parser_metadata_mode = 1;
      json_scanner_enter_kw_as_string_mode();
    }
    tokenizer_metadata_value
    { json_scanner_leave_kw_as_string_mode(); }
  | DECODER ':'
    {
      parser_metadata_mode = 3;
      json_scanner_enter_kw_as_string_mode();
    }
    tokenizer_metadata_value
    { json_scanner_leave_kw_as_string_mode(); }
  | NORMALIZER ':'
    {
      parser_metadata_mode = 2;
      json_scanner_enter_kw_as_string_mode();
    }
    tokenizer_metadata_value
    { json_scanner_leave_kw_as_string_mode(); }
  | STRING ':'
    { json_scanner_enter_kw_as_string_mode(); }
    json_value
    {
      free($1);
      json_scanner_leave_kw_as_string_mode();
    }
  ;

tokenizer_model_member_list
  : tokenizer_model_member_list ',' tokenizer_model_member
  | tokenizer_model_member
  ;

tokenizer_model_member
  : BPE_TYPE ':' STRING
    {
      if (strcmp($3, "BPE")) UTIL_ERROR("expected a BPE tokenizer model");
      free($3);
    }
  | VOCAB ':' '{'
    { json_scanner_enter_kw_as_string_mode(); }
    tokenizer_vocab_member_list '}'
    { json_scanner_leave_kw_as_string_mode(); }
  | MERGES ':' '['
    { json_scanner_enter_kw_as_string_mode(); }
    tokenizer_merge_list ']'
    { json_scanner_leave_kw_as_string_mode(); }
  | IGNORE_MERGES ':' BOOLEAN
    { parser_tokenizer->ignore_merges = $3; }
  | BYTE_FALLBACK ':' BOOLEAN
    { parser_tokenizer->metaspace.byte_fallback = $3; }
  | STRING ':'
    { json_scanner_enter_kw_as_string_mode(); }
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
      if (!$3.is_int || $3.ival < 0 ||
          $3.ival >= TOKENIZER_MAX_TOKEN_STRING) {
        yyerror("invalid vocabulary token id");
        YYABORT;
      }
      tokenizer_add_token(parser_tokenizer, $1, (int)$3.ival, false);
    }
  ;

tokenizer_merge_list
  : /* empty */
  | tokenizer_merges
  ;

tokenizer_merges
  : tokenizer_merges ',' tokenizer_merge
  | tokenizer_merge
  ;

tokenizer_merge
  : STRING { tokenizer_add_merge(parser_tokenizer, $1); }
  | '[' STRING ',' STRING ']'
    {
      size_t size = strlen($2) + strlen($4) + 2;
      char* pair = malloc(size);
      if (!pair) UTIL_DIE("failed to malloc for merge pair");
      snprintf(pair, size, "%s %s", $2, $4);
      tokenizer_add_merge(parser_tokenizer, pair);
      free($2);
      free($4);
    }
  ;

tokenizer_added_list
  : /* empty */
  | tokenizer_added_tokens
  ;

tokenizer_added_tokens
  : tokenizer_added_tokens ',' tokenizer_added
  | tokenizer_added
  ;

tokenizer_added
  : '{'
    {
      parser_added_id = -1;
      parser_added_string = NULL;
    }
    tokenizer_added_members '}'
    {
      if (parser_added_id < 0 || !parser_added_string) {
        yyerror("incomplete added token");
        YYABORT;
      }
      tokenizer_add_token(
          parser_tokenizer, parser_added_string, parser_added_id, true
      );
    }
  ;

tokenizer_added_members
  : tokenizer_added_members ',' tokenizer_added_member
  | tokenizer_added_member
  ;

tokenizer_added_member
  : STRING ':' STRING
    {
      if (!strcmp($1, "content")) {
        free(parser_added_string);
        parser_added_string = $3;
      } else free($3);
      free($1);
    }
  | STRING ':' NUMBER
    {
      if (!strcmp($1, "id")) {
        if (!$3.is_int || $3.ival < 0 ||
            $3.ival >= TOKENIZER_MAX_TOKEN_STRING) {
          yyerror("invalid added token id");
          YYABORT;
        }
        parser_added_id = (int)$3.ival;
      }
      free($1);
    }
  | STRING ':' BOOLEAN
    {
      if ($3 && (!strcmp($1, "single_word") || !strcmp($1, "lstrip") ||
                 !strcmp($1, "rstrip") || !strcmp($1, "normalized"))) {
        yyerror("unsupported added token matching option");
        YYABORT;
      }
      free($1);
    }
  ;

// Only retain the few pre-tokenizer/normalizer settings we implement.
tokenizer_metadata_value
  : STRING { free($1); }
  | NUMBER
  | BOOLEAN
  | NULL_
  | '{' tokenizer_metadata_members '}'
  | '[' tokenizer_metadata_values ']'
  ;

tokenizer_metadata_values
  : tokenizer_metadata_values ',' tokenizer_metadata_value
  | tokenizer_metadata_value
  | /* empty */
  ;

tokenizer_metadata_members
  : tokenizer_metadata_members ',' tokenizer_metadata_member
  | tokenizer_metadata_member
  | /* empty */
  ;

tokenizer_metadata_member
  : STRING ':' STRING { tokenizer_metadata_string($1, $3); }
  | STRING ':' BOOLEAN
    {
      if (parser_metadata_mode == 1 && $3 &&
          (!strcmp($1, "add_prefix_space") ||
                 !strcmp($1, "use_regex") || !strcmp($1, "invert"))) {
        yyerror("unsupported ByteLevel pre-tokenizer option");
        YYABORT;
      }
      if (parser_metadata_mode == 3) {
        parser_tokenizer->metaspace.decoder_invalid = true;
      }
      if (parser_metadata_mode == 1 && !strcmp($1, "split")) {
        parser_tokenizer->metaspace.unsplit = !$3;
      }
      free($1);
    }
  | STRING ':' NUMBER
    {
      if (parser_metadata_mode == 3) {
        if (!strcmp($1, "start") && $3.is_int && $3.ival == 1) {
          parser_tokenizer->metaspace.decoder_strip_start = true;
        } else if (!strcmp($1, "stop") && $3.is_int && $3.ival == 0) {
          parser_tokenizer->metaspace.decoder_strip_stop = true;
        } else {
          parser_tokenizer->metaspace.decoder_invalid = true;
        }
      }
      free($1);
    }
  | STRING ':' NULL_ { free($1); }
  | STRING ':' '{' tokenizer_metadata_members '}' { free($1); }
  | STRING ':' '[' tokenizer_metadata_values ']' { free($1); }
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
  parser_layer_count = 0;

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

  // Dense FFNs have zero counts; MoE selects 1..expert_count experts.
  size_t expert_count = parser_safetensors->expert_count;
  size_t selected_count = parser_safetensors->expert_per_token_count;
  if ((expert_count == 0 && selected_count != 0) ||
      (expert_count > 0 &&
       (selected_count == 0 || selected_count > expert_count))) {
    UTIL_DIE("invalid expert_count / expert_per_token_count combination");
  }

  if (parser_safetensors->rope_yarn &&
      (parser_safetensors->rope_factor < 1.0f ||
       parser_safetensors->rope_context_len == 0 ||
       parser_safetensors->rope_beta_fast <=
           parser_safetensors->rope_beta_slow)) {
    UTIL_DIE("invalid YaRN factor, context length or beta thresholds");
  }

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
  json_scanner_enter_tokenizer_mode();
  json_scanner_restart(json_scanner_in);
  yyparse();
  fclose(json_scanner_in);
  #ifdef DEBUG
  fprintf(stderr, "Done\n");
  #endif

  return parser_tokenizer;
}

// Match only the Replace -> ByteFallback -> Fuse -> Strip decoder chain.
static void metaspace_decoder_string(char* key, char* value) {
  tokenizer_metaspace_t* m = &parser_tokenizer->metaspace;
  const char* types[] = {
      "Sequence", "Replace", "ByteFallback", "Fuse", "Strip"
  };
  if (!strcmp(key, "type")) {
    if (m->decoder_type_count >= 5 ||
        strcmp(value, types[m->decoder_type_count])) {
      m->decoder_invalid = true;
    }
    m->decoder_type_count++;
  } else if (!strcmp(key, "String") && !strcmp(value, "\xe2\x96\x81")) {
    m->decoder_replace = true;
  } else if (!strcmp(key, "content") && !strcmp(value, " ")) {
    m->decoder_content_count++;
  } else {
    m->decoder_invalid = true;
  }
}

static void tokenizer_metadata_string(char* key, char* value) {
  if (parser_metadata_mode == 3) {
    metaspace_decoder_string(key, value);
  } else if (parser_metadata_mode == 1) {
    if (!strcmp(key, "replacement")) {
      parser_tokenizer->metaspace.replacement =
          !strcmp(value, "\xe2\x96\x81");
    } else if (!strcmp(key, "prepend_scheme")) {
      parser_tokenizer->metaspace.first = !strcmp(value, "first");
    } else if (!strcmp(key, "Regex")) {
      tokenizer_set_pattern(parser_tokenizer, value);
      value = NULL;
    } else if (!strcmp(key, "type")) {
      parser_tokenizer->metaspace.pre_tokenizer_count++;
      if (!strcmp(value, "Metaspace")) {
        parser_tokenizer->metaspace.enabled = true;
      } else if (!strcmp(value, "ByteLevel")) {
        parser_tokenizer->byte_level = true;
      } else if (strcmp(value, "Sequence") && strcmp(value, "Split")) {
        UTIL_ERROR("unsupported pre-tokenizer type");
      }
    } else if (!strcmp(key, "behavior") && strcmp(value, "Isolated")) {
      UTIL_ERROR("unsupported pre-tokenizer split behavior");
    }
  } else if (!strcmp(key, "type")) {
    if (strcmp(value, "NFC")) UTIL_ERROR("unsupported normalizer type");
    parser_tokenizer->normalize = true;
  }
  free(key);
  free(value);
}
