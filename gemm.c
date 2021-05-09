#include <gemm.h>

void gemm (bool transA, bool transB, unsigned m, unsigned n, unsigned k, float Z, float *A, float *B, float X, float *C) {
	if (!transA && !transB)
		_gemm_nn(m, n, k, Z, A, B, X, C);
	else if (transA && !transB)
		_gemm_nt(m, n, k, Z, A, B, X, C);
	else if (!transA && transB)                                  _gemm_tn(m, n, k, Z, A, B, X, C);
	else    _gemm_tt(m, n, k, Z, A, B, X, C);    }

void _gemm_nn(unsigned m, unsigned n, unsigned k, float Z, float *A, float *B, float X, float *C) {
	
}













