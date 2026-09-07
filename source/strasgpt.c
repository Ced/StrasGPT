#include "options.h"
#include "safetensors.h"
#include "sampler.h"
#include "tokenizer.h"
#include "transformer.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define _GNU_SOURCE
#include <sys/resource.h>

#ifdef PARALLEL
#include <mpi.h>
#include <omp.h>
#else
#define MPI_SUCCESS           0
#define MPI_COMM_WORLD        0
#define omp_get_thread_num()  0
#define omp_get_num_threads() 1
#define omp_set_num_threads(a)
#define MPI_Init(a, b)      MPI_SUCCESS
#define MPI_Comm_rank(a, b) (*(b) = 0, MPI_SUCCESS)
#define MPI_Comm_size(a, b) (*(b) = 1, MPI_SUCCESS)
#define MPI_Finalize()      MPI_SUCCESS
#endif

extern int json_scanner_lex_destroy(void);

// Global variables for MPI rank/size we may declare extern in other files
int mpi_rank, mpi_size;

extern unsigned long long total_more, total_zero;

// Return time in milliseconds, for benchmarking the model speed
static long time_in_ms(void) {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// Return peak memory usage of the process in GB
static double peak_rss_gb(void) {
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) != 0) {
    return -1.0;
  }

#if defined(__APPLE__) && defined(__MACH__)
  // macOS: ru_maxrss is in bytes
  return (double)ru.ru_maxrss / (1024.0 * 1024.0 * 1024.0);
#else
  // Linux: ru_maxrss is in kilobytes
  return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#endif
}

// Keep conversation history in the transformer cache, including turn endings.
static void chat(
    options_t* options,
    safetensors_t* safetensors,
    tokenizer_t* tokenizer,
    transformer_t* transformer,
    sampler_t* sampler
) {
  char* model_type = safetensors->model_type;
  char* first = "";
  char* prefix;
  char* suffix;
  char* ending;
  if (model_type && strstr(model_type, "qwen")) {
    prefix = "<|im_start|>user\n";
    suffix = "<|im_end|>\n<|im_start|>assistant\n";
    if (strstr(model_type, "qwen3_5")) {
      suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    ending = "<|im_end|>";
  } else if (model_type && strstr(model_type, "mistral")) {
    first = "<s>";
    prefix = "[INST]";
    suffix = "[/INST]";
    ending = "</s>";
  } else if (model_type && strstr(model_type, "llama")) {
    first = "<|begin_of_text|>";
    prefix = "<|start_header_id|>user<|end_header_id|>\n\n";
    suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    ending = "<|eot_id|>";
  } else if (model_type && strstr(model_type, "gpt_oss")) {
    first = "<|start|>system<|message|>You are a helpful assistant."
            "<|end|>";
    prefix = "<|start|>user<|message|>";
    suffix = "<|end|><|start|>assistant<|meta_sep|>final<|message|>";
    ending = "<|fim_suffix|>";
    // Older gpt-oss tokenizers use these names for the same delimiters.
    if (tokenizer->eos_token_id >= 0 &&
        strcmp(
            tokenizer->token_string[tokenizer->eos_token_id], "<|return|>"
        ) == 0) {
      suffix = "<|end|><|start|>assistant<|channel|>final<|message|>";
      ending = "<|return|>";
    }
  } else {
    UTIL_ERROR("--chat supports Qwen, Mistral, Llama 3 and gpt-oss models");
  }

  size_t ending_count = 0;
  int* ending_token = NULL;
  tokenizer_tokenize(
      tokenizer, ending, false, false, &ending_count, &ending_token
  );
  if (ending_count != 1 || !tokenizer->added[ending_token[0]]) {
    UTIL_ERROR("missing chat end-of-turn token");
  }

  size_t vocabulary_len;
  float* logits = transformer_logits_malloc(transformer, 1, &vocabulary_len);
  char* line = NULL;
  size_t line_size = 0;
  fprintf(stderr, "\nChat: one message per line; /quit or EOF to exit.\n");
  while (true) {
    fprintf(stderr, "\nYou: ");
    fflush(stderr);
    if (getline(&line, &line_size, stdin) < 0) {
      break;
    }
    line[strcspn(line, "\r\n")] = '\0';
    if (strcmp(line, "/quit") == 0) {
      break;
    }
    if (!line[0]) {
      continue;
    }

    bool first_turn = transformer->state->cached_count == 0;
    char* separator = !first_turn && strstr(model_type, "qwen") ? "\n" : "";
    size_t prompt_size = strlen(first) + strlen(separator) + strlen(prefix) +
                         strlen(line) + strlen(suffix) + 1;
    char* prompt = malloc(prompt_size);
    if (!prompt) {
      UTIL_DIE("malloc failed for chat prompt");
    }
    snprintf(
        prompt,
        prompt_size,
        "%s%s%s%s%s",
        first_turn ? first : "",
        separator,
        prefix,
        line,
        suffix
    );
    size_t token_count = 0;
    int* token = NULL;
    tokenizer_tokenize(tokenizer, prompt, false, false, &token_count, &token);
    free(prompt);

    // Reserve space for at least one reply token and the turn ending.
    size_t remaining_count =
        transformer->config->context_len - transformer->state->cached_count;
    if (remaining_count < 2 || token_count > remaining_count - 2) {
      free(token);
      fprintf(stderr, "\nChat context is full; start a new session.\n");
      break;
    }
    size_t reply_count =
        UTIL_MIN(options->step_count, remaining_count - token_count - 1);
    int predicted_token = token[token_count - 1];
    bool done = false;
    fprintf(stderr, "Assistant: ");
    fflush(stderr);
#pragma omp parallel shared(predicted_token, done)
    {
      transformer_predict(transformer, token_count, token, 1, logits);
      for (size_t i = 0; i < reply_count; i++) {
#pragma omp single
        {
          predicted_token = sampler_sample(sampler, logits, predicted_token);
          done = predicted_token == ending_token[0] ||
                 safetensors_is_eos(safetensors, predicted_token);
          if (!done) {
            tokenizer_print_token(stdout, tokenizer, predicted_token);
            fflush(stdout);
          }
        }
        if (done) {
          break;
        }
        // Consume even the last visible token so the next turn sees it.
        transformer_predict(transformer, 1, &predicted_token, 1, logits);
      }
      // Also close replies truncated by -n, without printing the delimiter.
      transformer_predict(transformer, 1, ending_token, 1, logits);
    }
    printf("\n");
    fflush(stdout);
    free(token);
  }
  free(line);
  free(logits);
  free(ending_token);
}

static void generate(
    options_t* options,
    safetensors_t* safetensors,
    tokenizer_t* tokenizer,
    transformer_t* transformer,
    sampler_t* sampler
) {
  // Get the prompt, either from file or command line argument
  char* prompt = NULL;
  char* file_prompt = NULL;
  if (options->use_prompt_file) {
    FILE* pf = fopen(options->prompt_file, "rb");
    if (!pf) {
      UTIL_ERROR("can't open prompt file");
    }
    fseek(pf, 0, SEEK_END);
    long fsize = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    file_prompt = malloc(fsize + 1);
    if (!file_prompt) {
      UTIL_DIE("malloc failed for file_prompt");
    }
    size_t read_count = fread(file_prompt, 1, fsize, pf);
    if (read_count != (size_t)fsize) {
      UTIL_ERROR("failed to read the entire prompt file");
    }
    fclose(pf);
    file_prompt[fsize] = '\0';
    // Strip trailing newline if present
    if (fsize > 0 && file_prompt[fsize - 1] == '\n') {
      file_prompt[fsize - 1] = '\0';
    }
    prompt = file_prompt;
  } else {
    prompt = options->prompt_string;
  }

  // Tokenize the prompt into token sequence
  size_t token_count = 0;
  int* token = NULL;
  if (options->pre_tokenized) {
    util_parse_tokens(
        prompt, &token_count, &token, false, tokenizer->bos_token_id
    );
  } else {
    tokenizer_tokenize(tokenizer, prompt, true, false, &token_count, &token);
  }
  if (token_count < 1) {
    UTIL_ERROR("expected at least 1 prompt token");
  }
  tokenizer_print_tokens(tokenizer, stderr, token_count, token, 4);
  fprintf(stderr, "\n");

  // Print the prompt string (in blue)
  fprintf(stderr, "\033[1;34m");
  for (size_t i = 0; i < token_count; i++) {
    tokenizer_print_token(stderr, tokenizer, token[i]);
  }
  fprintf(stderr, "\033[0m");

  // Prepare timing
  long start = 0;
  long end = 0;
  double prefill_time = 0.0;
  double decode_time = 0.0;

  // Prepare to get prediction results
  size_t generated_count = 0; // Number of tokens generated so far
  size_t vocabulary_len = 0;  // Will be filled by transformer_logits_malloc
  float* logits = transformer_logits_malloc(transformer, 1, &vocabulary_len);
  int predicted_token = 0;
  start = time_in_ms();
  bool continue_generation = true;

#pragma omp parallel shared(continue_generation)
  {
    // First achieve prompt processing (prefill):
    // - Get the logits (probability distribution) for the next token
    transformer_predict(transformer, token_count, token, 1, logits);

#pragma omp single
    {
      end = time_in_ms();
      prefill_time = (end - start) / 1000.0;
      // - Select the next token from the logits (last token for penalty)
      predicted_token = sampler_sample(sampler, logits, token[token_count - 1]);
      // - Print the decoded token bytes
      tokenizer_print_token(stdout, tokenizer, predicted_token);
      generated_count++;
      continue_generation = !safetensors_is_eos(safetensors, predicted_token);

      start = time_in_ms();
    }

    // Then achieve token generation (decode), one by one
    while (generated_count < options->step_count && continue_generation) {
      transformer_predict(transformer, 1, &predicted_token, 1, logits);

#pragma omp single
      {
        end = time_in_ms();
        decode_time += (end - start) / 1000.0;

        predicted_token = sampler_sample(sampler, logits, predicted_token);
        generated_count++;
        if (!safetensors_is_eos(safetensors, predicted_token)) {
          tokenizer_print_token(stdout, tokenizer, predicted_token);
          // If we want to dump token ids
          // fprintf(stdout, "%d ", predicted_token);
        } else {
          // End of string token, stop generating (set loop exit condition)
          tokenizer_print_token(stdout, tokenizer, predicted_token);
          continue_generation = false;
        }
        start = time_in_ms();
      }
    }
  }
  printf("\n");

  // Report maximum memory usage by the process
  fprintf(stderr, "\nMax memory used (RSS): %.2f GB", peak_rss_gb());

  // Report achieved tok/s
  fprintf(
      stderr,
      "\nPrompt processing (prefill): %4zu tokens in %7.3f s (%f token/s)\n",
      token_count,
      prefill_time,
      token_count / prefill_time
  );
  if (generated_count > 1) {
    fprintf(
        stderr,
        "Token generation  (decode):  %4zu tokens in %7.3f s (%f token/s)\n",
        (generated_count - 1),
        decode_time,
        (generated_count - 1) / decode_time
    );
  }

  // Cleanup
  if (options->use_prompt_file) {
    free(file_prompt);
  }
  free(logits);
  free(token);
}

int main(int argc, char* argv[]) {
  // Prepare all the components needed for text generation:
  // - Options from command line arguments
  options_t* options = options_read(argc, argv);
  options_print(stderr, options);
  fprintf(stderr, "\n");

  // Let all processes and threads print their IDs
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    UTIL_DIE("MPI_Init failed");
  }
  if (MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS) {
    UTIL_DIE("MPI_Comm_rank failed");
  }
  if (MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS) {
    UTIL_DIE("MPI_Comm_size failed");
  }

  if (options->chat && mpi_size != 1) {
    UTIL_ERROR("--chat requires a single MPI rank (threads are supported)");
  }

  // Set OpenMP parameters
  // Let chat workers sleep while waiting for user input instead of spinning.
  setenv("OMP_WAIT_POLICY", options->chat ? "PASSIVE" : "ACTIVE", 1);
  omp_set_num_threads(options->thread_count);

#pragma omp parallel
  fprintf(
      stderr,
      "StrasGPT OpenMP thread %2d (total %2d) of MPI rank %2d (total %2d)\n",
      omp_get_thread_num(),
      omp_get_num_threads(),
      mpi_rank,
      mpi_size
  );

  if (mpi_rank == 0) {
    fprintf(stderr, "\n");
  }

  // - Safetensors model files
  safetensors_t* safetensors = safetensors_read(options);
  if (options->show_safetensors) {
    safetensors_print(stderr, safetensors);
    safetensors_free(safetensors);
    options_free(options);
    return EXIT_SUCCESS;
  }

  if (options->show_model) {
    safetensors_print_model_infos(stderr, safetensors);
    safetensors_free(safetensors);
    options_free(options);
    return EXIT_SUCCESS;
  }

  // - Tokenizer (ugly getting EOS/BOS from safetensors at the moment)
  tokenizer_t* tokenizer = tokenizer_read(options);
  tokenizer->bos_token_id = safetensors->bos_token_id;
  tokenizer->eos_token_id =
      safetensors->eos_token_count ? safetensors->eos_token_id[0] : -1;
  tokenizer_print(stderr, tokenizer);
  fprintf(stderr, "\n");

  // - Transformer model from safetensors
  transformer_t* transformer = transformer_from_safetensors(safetensors);
  transformer_print(stderr, transformer);
  fprintf(stderr, "\n");

  // - Sampler for next-token selection
  sampler_t* sampler = sampler_build(options, transformer);
  sampler_print(stderr, sampler);
  fprintf(stderr, "\n");
#ifdef DEBUG
  sampler->tokenizer = tokenizer; // For debug prints
#endif

  if (options->chat) {
    chat(options, safetensors, tokenizer, transformer, sampler);
  } else {
    generate(options, safetensors, tokenizer, transformer, sampler);
  }

  // Cleanup
  safetensors_free(safetensors);
  tokenizer_free(tokenizer);
  transformer_free(transformer);
  sampler_free(sampler);
  options_free(options);
  json_scanner_lex_destroy();

  if (MPI_Finalize() != MPI_SUCCESS) {
    UTIL_DIE("MPI_Finalize failed");
  }
  return EXIT_SUCCESS;
}
