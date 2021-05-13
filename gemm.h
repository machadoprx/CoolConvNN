#include <stdbool.h>
#include "utils.h"

void gemm (bool transA, bool transB, unsigned m, unsigned n, unsigned k, float *A, float *B, float X, float *C);