#ifndef utils_h
#define utils_h

#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CACHE_LINE 64
#define aalloc(bytes) aligned_alloc(CACHE_LINE, bytes);

enum layer_t{CONV, FC, MAX_POOL, ACTIVATION, BN};
enum activ_t{NONE, RELU};

void get_gauss(float mean, float stddev, float *u, float *v);
int dec_activation(const char* name);
int dec_layer_type(const char* name);

#endif