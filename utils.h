#ifndef utils_h
#define utils_h

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define CACHE_LINE 32
#define LINUX 1

#define f_apply(type, f, ...) {									\
	void *stop_loop = (int[]){0};								\
	type **list = (type*[]){__VA_ARGS__, stop_loop};			\
	for (int i = 0; list[i] != stop_loop; i++)					\
		f(list[i]);												\
}

void get_gauss(float mean, float stddev, float *u, float *v);

#endif