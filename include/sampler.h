#ifndef SAMPLER_H
# define SAMPLER_H

#include <stddef.h>
#include <stdio.h>

struct options;
struct transformer;

typedef struct {
  float probability;
  size_t index;
} sampler_probability_index_t;

typedef struct {
  size_t vocabulary_len;
  float temperature;
  float top_p;
  unsigned long long rng_state;
  sampler_probability_index_t* probindex; // Buffer used in top-p sampling
} sampler_t;

sampler_t* sampler_build(struct options* o, struct transformer* t);
void sampler_free(sampler_t* sampler);
void sampler_print(FILE* f, const sampler_t* sampler);
size_t sampler_sample(sampler_t* sampler, float* logits);

#endif // SAMPLER_H
