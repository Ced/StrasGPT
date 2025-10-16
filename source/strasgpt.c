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
#include <omp.h>
#include <mpi.h>
#else
#define MPI_SUCCESS 0
#define MPI_COMM_WORLD 0
#define omp_get_thread_num()  0
#define omp_get_num_threads() 1
#define MPI_Init(a, b)        MPI_SUCCESS
#define MPI_Comm_rank(a, b)   (*(b) = 0, MPI_SUCCESS)
#define MPI_Comm_size(a, b)   (*(b) = 1, MPI_SUCCESS)
#define MPI_Finalize()        MPI_SUCCESS
#endif

extern int json_scanner_lex_destroy(void);

// Global variables for MPI rank/size we may declare extern in other files
int mpi_rank, mpi_size;

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

int main(int argc, char* argv[]) {
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

  // Prepare all the components needed for text generation:
  // - Options from command line arguments
  options_t* options = options_read(argc, argv);
  options_print(stderr, options);
  fprintf(stderr, "\n");

  // - Safetensors model files
  safetensors_t* safetensors = safetensors_read(options);
  #ifdef DEBUG
  safetensors_print(stderr, safetensors);
  fprintf(stderr, "\n");
  #endif

  // - Tokenizer
  tokenizer_t* tokenizer = tokenizer_read(options);
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
  tokenizer_tokenize(tokenizer, prompt, true, false, &token_count, &token);
  if (token_count < 1) {
    UTIL_ERROR("expected at least 1 prompt token");
  }
  tokenizer_print_tokens(tokenizer, stderr, token_count, token, 4);
  fprintf(stderr, "\n");

  // Print the prompt string (in blue)
  fprintf(stderr, "\033[1;34m");
  while (*prompt) {
    fprintf(stderr, "%c", *prompt);
    prompt++;
  }
  fprintf(stderr, "\033[0m");

  // Prepare timing
  long start = 0;
  long end = 0;
  double prefill_time = 0.0;
  double decode_time = 0.0;

  // Prepare to get prediction results
  size_t generated_count = 0; // Number of tokens generated so far
  size_t vocabulary_len = 0; // Will be filled by transformer_logits_malloc
  float* logits = transformer_logits_malloc(transformer, 1, &vocabulary_len);

  // First achieve prompt processing (prefill):
  // - Get the logits (probability distribution) for the next token
  start = time_in_ms();
  transformer_predict(transformer, token_count, token, 1, logits);
  end = time_in_ms();
  prefill_time = (end - start) / 1000.0;
  // - Select the next token from the logits
  int predicted_token = sampler_sample(sampler, logits);
  // - Decode the token into a string
  char* predicted_string = tokenizer_decode(tokenizer, predicted_token);
  // - Print the token string
  tokenizer_print_token_string(stdout, predicted_string);
  generated_count++;

  // Then achieve token generation (decode), one by one
  while (generated_count < options->step_count) {
    start = time_in_ms();
    transformer_predict(transformer, 1, &predicted_token, 1, logits);
    end = time_in_ms();
    decode_time += (end - start) / 1000.0;

    predicted_token = sampler_sample(sampler, logits);
    generated_count++;
    if (predicted_token != TOKENIZER_TOKEN_EOS) {
      predicted_string = tokenizer_decode(tokenizer, predicted_token);
      tokenizer_print_token_string(stdout, predicted_string);
    } else {
      // End of string token, stop generating
      fprintf(stdout, "%s", TOKENIZER_STRING_TOKEN_EOS);
      break;
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
  safetensors_free(safetensors);
  tokenizer_free(tokenizer);
  transformer_free(transformer);
  sampler_free(sampler);
  options_free(options);
  free(logits);
  free(token);
  json_scanner_lex_destroy();

  if (MPI_Finalize() != MPI_SUCCESS) {
    UTIL_DIE("MPI_Finalize failed");
  }

  return EXIT_SUCCESS;
}
