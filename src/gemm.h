#ifndef MINNET_GEMM_H
#define MINNET_GEMM_H

void mn_gemm(int ta, int tb, int m, int n, int k,
             const float *a, const float *b, float beta, float *c);

#endif
