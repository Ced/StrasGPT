#ifndef OPTIONS_H
# define options_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define OPTIONS_DEFAULT_PROMPT_FILE      "prompt.txt"
#define OPTIONS_DEFAULT_PROMPT_STRING    "Once upon a time"
#define OPTIONS_DEFAULT_MODEL_DIR        "."
#define OPTIONS_DEFAULT_STEP_COUNT       256
#define OPTIONS_DEFAULT_THREAD           1
#define OPTIONS_DEFAULT_PRESENCE_PENALTY 0.0f
#define OPTIONS_DEFAULT_TEMPERATURE      1.0f
#define OPTIONS_DEFAULT_TOP_K            40
#define OPTIONS_DEFAULT_TOP_P            0.9f

typedef struct options {
  bool use_prompt_file;
  char* prompt_file;
  char* prompt_string;
  bool pre_tokenized;
  char* model_dir;
  size_t step_count;
  size_t thread_count;
  float presence_penalty;
  float temperature;
  size_t top_k;
  float top_p;
  bool seed_is_set;
  unsigned long long seed;
  bool show_model;
  bool show_safetensors;
} options_t;

options_t* options_malloc(void);
void options_free(options_t* options);
void options_print(FILE* f, const options_t* options);
options_t* options_read(int argc, char* argv[]);

#endif // OPTIONS_H
