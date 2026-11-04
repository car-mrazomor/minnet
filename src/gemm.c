#include "gemm.h"

#if defined(MN_ACCELERATE)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#define MN_BLAS
#elif defined(MN_CBLAS)
#include <cblas.h>
#define MN_BLAS
#endif

void mn_gemm(int ta, int tb, int m, int n, int k,
             const float *a, const float *b, float beta, float *c)
{
#ifdef MN_BLAS
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                m, n, k, 1.0f, a, ta ? m : k, b, tb ? k : n, beta, c, n);
#else
    for (int i = 0; i < m; i++) {
        float *cr = c + (long)i * n;
        for (int j = 0; j < n; j++) cr[j] = beta == 0.0f ? 0.0f : cr[j] * beta;
        for (int p = 0; p < k; p++) {
            float av = ta ? a[(long)p * m + i] : a[(long)i * k + p];
            if (av == 0.0f) continue;
            if (tb) {
                for (int j = 0; j < n; j++) cr[j] += av * b[(long)j * k + p];
            } else {
                const float *br = b + (long)p * n;
                for (int j = 0; j < n; j++) cr[j] += av * br[j];
            }
        }
    }
#endif
}
