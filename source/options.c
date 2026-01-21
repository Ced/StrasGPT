#include "options.h"
#include "util.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Allocate an options_t structure and set defaults
options_t* options_malloc(void) {
  options_t* options = calloc(1, sizeof(options_t));
  if (!options) {
    UTIL_DIE("failed to malloc for options_t");
  }
  options->use_prompt_file = false;
  options->prompt_file = OPTIONS_DEFAULT_PROMPT_FILE;
  options->prompt_string = OPTIONS_DEFAULT_PROMPT_STRING;
  options->pre_tokenized = false;
  options->model_dir = OPTIONS_DEFAULT_MODEL_DIR;
  options->step_count = OPTIONS_DEFAULT_STEP_COUNT;
  options->thread_count = OPTIONS_DEFAULT_THREAD;
  options->presence_penalty = OPTIONS_DEFAULT_PRESENCE_PENALTY;
  options->temperature = OPTIONS_DEFAULT_TEMPERATURE;
  options->top_k = OPTIONS_DEFAULT_TOP_K;
  options->top_p = OPTIONS_DEFAULT_TOP_P;
  options->seed_is_set = false;
  options->show_model = false;
  options->show_safetensors = false;
  options->instruct = false;
  return options;
}

// Free an options_t structure
void options_free(options_t* options) {
  free(options);
}

// Print an options_t structure
void options_print(FILE* f, const options_t* options) {
  if (!options) {
    fprintf(f, "Options: NULL\n");
    return;
  }

  // Print options
  fprintf(f, "Options:\n");
  if (options->use_prompt_file) {
    fprintf(f, "- prompt (file):    %s\n", options->prompt_file);
  } else {
    fprintf(f, "- prompt (string):  %s\n", options->prompt_string);
  }
  fprintf(
      f, "- pre_tokenized:    %s\n", options->pre_tokenized ? "true" : "false"
  );
  fprintf(f, "- model_dir:        %s\n", options->model_dir);
  fprintf(f, "- step_count:       %zu\n", options->step_count);
  fprintf(f, "- thread_count:     %zu\n", options->thread_count);
  fprintf(f, "- presence_penalty: %.3f\n", options->presence_penalty);
  fprintf(f, "- temperature:      %.3f\n", options->temperature);
  fprintf(f, "- top_k:            %zu\n", options->top_k);
  fprintf(f, "- top_p:            %.3f\n", options->top_p);
  if (options->seed_is_set) {
    fprintf(f, "- seed:             %llu\n", options->seed);
  } else {
    fprintf(f, "- seed:             (not set)\n");
  }
  fprintf(
      f, "- show_model:       %s\n", options->show_model ? "true" : "false"
  );
  fprintf(
      f,
      "- show_safetensors: %s\n",
      options->show_safetensors ? "true" : "false"
  );
  fprintf(f, "- instruct:         %s\n", options->instruct ? "true" : "false");
}

void option_usage(char* argv[], int status) {
  fprintf(stderr, "Usage: %s [options]\n", argv[0]);
  fprintf(stderr, "Example: %s -m model_dir -n 32 -p \"Once upon a\"\n", argv[0]);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -f <path>       read prompt from file\n");
  fprintf(stderr, "  -h, --help      print usage and exit\n");
  fprintf(stderr, "  --instruct      wrap prompt with instruction tokens\n");
  fprintf(stderr, "  -m <dir>        model directory (default ./)\n");
  fprintf(stderr, "  -n <int>        num of tokens to predict (default 256)\n");
  fprintf(stderr, "  -p <string>     input prompt\n");
  fprintf(stderr, "  --presence-penalty <float> presence penalty in [0, 2]\n");
  fprintf(stderr, "                  (default 0.0)\n");
  fprintf(stderr, "  --pre-tokenized input pre-tokenized (space separated)\n");
  fprintf(stderr, "  -s <int>        random seed (default time(NULL))\n");
  fprintf(stderr, "  --show-model    show model infos and exit\n");
  fprintf(stderr, "  --show-safetensors show safetensors infos and exit\n");
  fprintf(stderr, "  -t <int>        set the number of threads (default 1)\n");
  fprintf(stderr, "  --temp <float>  temperature in [0, inf] (default 1.0)\n");
  fprintf(stderr, "  --top-k <int>   top-k sampling (0: none, default 40)\n");
  fprintf(stderr, "  --top-p <float> top-p sampling in [0, 1] (default 0.9)\n");
  exit(status);
}

// Parse command line arguments into an options_t structure
options_t* options_read(int argc, char* argv[]) {
  options_t* options = options_malloc();
  size_t arg_count = (size_t)argc;

  for (size_t i = 1; i < arg_count; i += 2) {
    if (strcmp(argv[i], "-f") == 0) {
      if (i + 1 < arg_count) {
        options->prompt_file = argv[i + 1];
        options->use_prompt_file = true;
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      options_free(options);
      option_usage(argv, EXIT_SUCCESS);
    } else if (strcmp(argv[i], "--instruct") == 0) {
      options->instruct = true;
      i--;
    } else if (strcmp(argv[i], "-m") == 0) {
      if (i + 1 < arg_count) {
        options->model_dir = argv[i + 1];
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "-n") == 0) {
      if (i + 1 < arg_count) {
        options->step_count = strtol(argv[i + 1], NULL, 10);
        if (options->step_count < 1) {
          options_free(options);
          option_usage(argv, EXIT_FAILURE);
        }
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "-p") == 0) {
      if (i + 1 < arg_count) {
        options->prompt_string = argv[i + 1];
        options->use_prompt_file = false;
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "--presence-penalty") == 0) {
      if (i + 1 < arg_count) {
        options->presence_penalty = strtod(argv[i + 1], NULL);
        if (options->presence_penalty < 0.0 ||
            options->presence_penalty > 2.0) {
          options_free(options);
          option_usage(argv, EXIT_FAILURE);
        }
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "--pre-tokenized") == 0) {
      options->pre_tokenized = true;
      i--;
    } else if (strcmp(argv[i], "-s") == 0) {
      if (i + 1 < arg_count) {
        options->seed = strtoull(argv[i + 1], NULL, 10);
        options->seed_is_set = true;
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "--show-model") == 0) {
      options->show_model = true;
      i--;
    } else if (strcmp(argv[i], "--show-safetensors") == 0) {
      options->show_safetensors = true;
      i--;
    } else if (strcmp(argv[i], "-t") == 0) {
      if (i + 1 < arg_count) {
        int thread_count = strtoll(argv[i + 1], NULL, 10);
        if (thread_count <= 0) {
          options_free(options);
          UTIL_ERROR("Invalid number of threads");
        }
        options->thread_count = thread_count;
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "--temp") == 0) {
      if (i + 1 < arg_count) {
        options->temperature = strtod(argv[i + 1], NULL);
        if (options->temperature < 0.0) {
          options_free(options);
          option_usage(argv, EXIT_FAILURE);
        }
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "--top-k") == 0) {
      if (i + 1 < arg_count) {
        options->top_k = strtol(argv[i + 1], NULL, 10);
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else if (strcmp(argv[i], "--top-p") == 0) {
      if (i + 1 < arg_count) {
        options->top_p = strtod(argv[i + 1], NULL);
        if (options->top_p < 0 || options->top_p > 1) {
          options_free(options);
          option_usage(argv, EXIT_FAILURE);
        }
      } else {
        options_free(options);
        option_usage(argv, EXIT_FAILURE);
      }
    } else {
      options_free(options);
      option_usage(argv, EXIT_FAILURE);
    }
  }

  return options;
}
