#include "minnet.h"
#include "gemm.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { L_CONV = 1, L_DENSE = 2, L_RELU = 3, L_MAXPOOL = 4 };

typedef struct {
    int kind;
    int ic, ih, iw;
    int oc, oh, ow;
    int size, stride, pad;
    int nw, np;
    float *p, *g, *m, *v;
    float *out, *delta;
    int *idx;
} layer;

struct mn_net {
    int c, h, w;
    int count, n, cap, t;
    layer *layers;
    float *mean, *std;
    float *in, *dout, *col, *dcol;
    size_t colsize;
};

#define GROW(ptr, count)                                              \
    do {                                                              \
        void *q_ = realloc((ptr), (size_t)(count) * sizeof *(ptr));   \
        if (!q_) return -1;                                           \
        (ptr) = q_;                                                   \
    } while (0)

static const char MAGIC[4] = {'M', 'N', 'E', 'T'};

static int isize(const layer *l) { return l->ic * l->ih * l->iw; }
static int osize(const layer *l) { return l->oc * l->oh * l->ow; }

static float uniform(uint64_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return ((float)(*s >> 40) + 0.5f) / 16777216.0f;
}

static float gauss(uint64_t *s)
{
    float u = uniform(s), v = uniform(s);
    return sqrtf(-2.0f * logf(u)) * cosf(6.28318531f * v);
}

mn_net *mn_new(int c, int h, int w)
{
    if (c <= 0 || h <= 0 || w <= 0 || (long long)c * h * w > INT_MAX) return NULL;
    mn_net *net = calloc(1, sizeof *net);
    if (!net) return NULL;
    net->c = c;
    net->h = h;
    net->w = w;
    net->mean = calloc(c, sizeof(float));
    net->std = malloc(c * sizeof(float));
    if (!net->mean || !net->std) {
        mn_free(net);
        return NULL;
    }
    for (int i = 0; i < c; i++) net->std[i] = 1.0f;
    return net;
}

static void free_layer(layer *l)
{
    free(l->p);
    free(l->g);
    free(l->m);
    free(l->v);
    free(l->out);
    free(l->delta);
    free(l->idx);
}

void mn_free(mn_net *net)
{
    if (!net) return;
    for (int i = 0; i < net->count; i++) free_layer(&net->layers[i]);
    free(net->layers);
    free(net->mean);
    free(net->std);
    free(net->in);
    free(net->dout);
    free(net->col);
    free(net->dcol);
    free(net);
}

static layer *push(mn_net *net, int kind)
{
    if (!net) return NULL;
    layer *ls = realloc(net->layers, (size_t)(net->count + 1) * sizeof *ls);
    if (!ls) return NULL;
    net->layers = ls;
    layer *l = &ls[net->count];
    memset(l, 0, sizeof *l);
    l->kind = kind;
    if (net->count) {
        const layer *p = &ls[net->count - 1];
        l->ic = p->oc;
        l->ih = p->oh;
        l->iw = p->ow;
    } else {
        l->ic = net->c;
        l->ih = net->h;
        l->iw = net->w;
    }
    return l;
}

static int commit(mn_net *net, layer *l, long long np, long long nw)
{
    if (l->oc <= 0 || l->oh <= 0 || l->ow <= 0) return -1;
    if ((long long)l->oc * l->oh * l->ow > INT_MAX || np > INT_MAX) return -1;
    if (np) {
        l->p = calloc(np, sizeof(float));
        l->g = calloc(np, sizeof(float));
        l->m = calloc(np, sizeof(float));
        l->v = calloc(np, sizeof(float));
        if (!l->p || !l->g || !l->m || !l->v) {
            free_layer(l);
            return -1;
        }
    }
    l->np = (int)np;
    l->nw = (int)nw;
    net->count++;
    net->cap = 0;
    return 0;
}

int mn_conv(mn_net *net, int filters, int size, int stride, int pad)
{
    if (filters <= 0 || size <= 0 || stride <= 0 || pad < 0) return -1;
    layer *l = push(net, L_CONV);
    if (!l) return -1;
    if (l->ih + 2 * pad < size || l->iw + 2 * pad < size) return -1;
    l->size = size;
    l->stride = stride;
    l->pad = pad;
    l->oc = filters;
    l->oh = (l->ih + 2 * pad - size) / stride + 1;
    l->ow = (l->iw + 2 * pad - size) / stride + 1;
    long long nw = (long long)filters * l->ic * size * size;
    size_t cs = (size_t)l->ic * size * size * l->oh * l->ow;
    if (commit(net, l, nw + filters, nw)) return -1;
    if (cs > net->colsize) net->colsize = cs;
    return 0;
}

int mn_dense(mn_net *net, int outputs)
{
    if (outputs <= 0) return -1;
    layer *l = push(net, L_DENSE);
    if (!l) return -1;
    l->oc = outputs;
    l->oh = l->ow = 1;
    long long nw = (long long)outputs * isize(l);
    return commit(net, l, nw + outputs, nw);
}

int mn_relu(mn_net *net)
{
    layer *l = push(net, L_RELU);
    if (!l) return -1;
    l->oc = l->ic;
    l->oh = l->ih;
    l->ow = l->iw;
    return commit(net, l, 0, 0);
}

int mn_maxpool(mn_net *net, int size, int stride)
{
    if (size <= 0 || stride <= 0) return -1;
    layer *l = push(net, L_MAXPOOL);
    if (!l) return -1;
    if (l->ih < size || l->iw < size) return -1;
    l->size = size;
    l->stride = stride;
    l->oc = l->ic;
    l->oh = (l->ih - size) / stride + 1;
    l->ow = (l->iw - size) / stride + 1;
    return commit(net, l, 0, 0);
}

void mn_init(mn_net *net, unsigned seed)
{
    uint64_t s = ((uint64_t)seed + 1) * 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < net->count; i++) {
        layer *l = &net->layers[i];
        if (!l->np) continue;
        float sd = sqrtf(2.0f * l->oc / (float)l->nw);
        for (int j = 0; j < l->nw; j++) l->p[j] = sd * gauss(&s);
        for (int j = l->nw; j < l->np; j++) l->p[j] = 0.0f;
        memset(l->m, 0, l->np * sizeof(float));
        memset(l->v, 0, l->np * sizeof(float));
    }
    net->t = 0;
}

void mn_set_norm(mn_net *net, const float *mean, const float *std)
{
    for (int i = 0; i < net->c; i++) {
        net->mean[i] = mean[i];
        net->std[i] = std[i] > 0.0f ? std[i] : 1.0f;
    }
}

void mn_input_shape(const mn_net *net, int *c, int *h, int *w)
{
    *c = net->c;
    *h = net->h;
    *w = net->w;
}

int mn_input_size(const mn_net *net) { return net->c * net->h * net->w; }

int mn_output_size(const mn_net *net)
{
    return net->count ? osize(&net->layers[net->count - 1]) : mn_input_size(net);
}

int mn_layer_count(const mn_net *net) { return net->count; }

int mn_param_count(const mn_net *net, int i)
{
    return i >= 0 && i < net->count ? net->layers[i].np : 0;
}

float *mn_params(mn_net *net, int i)
{
    return i >= 0 && i < net->count ? net->layers[i].p : NULL;
}

float *mn_grads(mn_net *net, int i)
{
    return i >= 0 && i < net->count ? net->layers[i].g : NULL;
}

static int ensure(mn_net *net, int n)
{
    if (n <= net->cap) return 0;
    GROW(net->in, (size_t)n * mn_input_size(net));
    GROW(net->dout, (size_t)n * mn_output_size(net));
    for (int i = 0; i < net->count; i++) {
        layer *l = &net->layers[i];
        GROW(l->out, (size_t)n * osize(l));
        GROW(l->delta, (size_t)n * isize(l));
        if (l->kind == L_MAXPOOL) GROW(l->idx, (size_t)n * osize(l));
    }
    if (net->colsize) {
        GROW(net->col, net->colsize);
        GROW(net->dcol, net->colsize);
    }
    net->cap = n;
    return 0;
}

static void im2col(const float *im, const layer *l, float *col)
{
    int k = l->size, s = l->stride, p = l->pad, P = l->oh * l->ow;
    for (int c = 0; c < l->ic; c++) {
        const float *src = im + (size_t)c * l->ih * l->iw;
        for (int ky = 0; ky < k; ky++)
            for (int kx = 0; kx < k; kx++) {
                float *dst = col + (size_t)((c * k + ky) * k + kx) * P;
                for (int y = 0; y < l->oh; y++) {
                    int iy = y * s - p + ky;
                    float *row = dst + y * l->ow;
                    if (iy < 0 || iy >= l->ih) {
                        memset(row, 0, l->ow * sizeof(float));
                        continue;
                    }
                    const float *sr = src + iy * l->iw;
                    for (int x = 0; x < l->ow; x++) {
                        int ix = x * s - p + kx;
                        row[x] = ix >= 0 && ix < l->iw ? sr[ix] : 0.0f;
                    }
                }
            }
    }
}

static void col2im(const float *col, const layer *l, float *im)
{
    int k = l->size, s = l->stride, p = l->pad, P = l->oh * l->ow;
    for (int c = 0; c < l->ic; c++) {
        float *dst = im + (size_t)c * l->ih * l->iw;
        for (int ky = 0; ky < k; ky++)
            for (int kx = 0; kx < k; kx++) {
                const float *src = col + (size_t)((c * k + ky) * k + kx) * P;
                for (int y = 0; y < l->oh; y++) {
                    int iy = y * s - p + ky;
                    if (iy < 0 || iy >= l->ih) continue;
                    const float *row = src + y * l->ow;
                    float *dr = dst + iy * l->iw;
                    for (int x = 0; x < l->ow; x++) {
                        int ix = x * s - p + kx;
                        if (ix >= 0 && ix < l->iw) dr[ix] += row[x];
                    }
                }
            }
    }
}

static void conv_forward(mn_net *net, layer *l, const float *in, int n)
{
    int K = l->ic * l->size * l->size, P = l->oh * l->ow;
    for (int b = 0; b < n; b++) {
        float *o = l->out + (size_t)b * osize(l);
        im2col(in + (size_t)b * isize(l), l, net->col);
        for (int f = 0; f < l->oc; f++) {
            float bias = l->p[l->nw + f];
            for (int i = 0; i < P; i++) o[f * P + i] = bias;
        }
        mn_gemm(0, 0, l->oc, P, K, l->p, net->col, 1.0f, o);
    }
}

static void conv_backward(mn_net *net, layer *l, const float *in, const float *d, int n, int need)
{
    int K = l->ic * l->size * l->size, P = l->oh * l->ow;
    for (int b = 0; b < n; b++) {
        const float *db = d + (size_t)b * osize(l);
        im2col(in + (size_t)b * isize(l), l, net->col);
        mn_gemm(0, 1, l->oc, K, P, db, net->col, 1.0f, l->g);
        for (int f = 0; f < l->oc; f++) {
            float sum = 0.0f;
            for (int i = 0; i < P; i++) sum += db[f * P + i];
            l->g[l->nw + f] += sum;
        }
        if (need) {
            float *dx = l->delta + (size_t)b * isize(l);
            mn_gemm(1, 0, K, P, l->oc, l->p, db, 0.0f, net->dcol);
            memset(dx, 0, isize(l) * sizeof(float));
            col2im(net->dcol, l, dx);
        }
    }
}

static void dense_forward(layer *l, const float *in, int n)
{
    int I = isize(l), O = l->oc;
    for (int b = 0; b < n; b++)
        memcpy(l->out + (size_t)b * O, l->p + l->nw, O * sizeof(float));
    mn_gemm(0, 1, n, O, I, in, l->p, 1.0f, l->out);
}

static void dense_backward(layer *l, const float *in, const float *d, int n, int need)
{
    int I = isize(l), O = l->oc;
    mn_gemm(1, 0, O, I, n, d, in, 1.0f, l->g);
    for (int b = 0; b < n; b++)
        for (int j = 0; j < O; j++) l->g[l->nw + j] += d[(size_t)b * O + j];
    if (need) mn_gemm(0, 0, n, I, O, d, l->p, 0.0f, l->delta);
}

static void relu_forward(layer *l, const float *in, int n)
{
    size_t total = (size_t)n * osize(l);
    for (size_t i = 0; i < total; i++) l->out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

static void relu_backward(layer *l, const float *d, int n)
{
    size_t total = (size_t)n * osize(l);
    for (size_t i = 0; i < total; i++) l->delta[i] = l->out[i] > 0.0f ? d[i] : 0.0f;
}

static void pool_forward(layer *l, const float *in, int n)
{
    int k = l->size, s = l->stride, hw = l->ih * l->iw;
    for (int b = 0; b < n; b++)
        for (int c = 0; c < l->ic; c++) {
            const float *src = in + ((size_t)b * l->ic + c) * hw;
            for (int y = 0; y < l->oh; y++)
                for (int x = 0; x < l->ow; x++) {
                    float best = -INFINITY;
                    int bi = (y * s) * l->iw + x * s;
                    for (int ky = 0; ky < k; ky++)
                        for (int kx = 0; kx < k; kx++) {
                            int i = (y * s + ky) * l->iw + x * s + kx;
                            if (src[i] > best) {
                                best = src[i];
                                bi = i;
                            }
                        }
                    size_t o = (((size_t)b * l->oc + c) * l->oh + y) * l->ow + x;
                    l->out[o] = best;
                    l->idx[o] = c * hw + bi;
                }
        }
}

static void pool_backward(layer *l, const float *d, int n)
{
    int is = isize(l), os = osize(l);
    memset(l->delta, 0, (size_t)n * is * sizeof(float));
    for (int b = 0; b < n; b++) {
        float *dx = l->delta + (size_t)b * is;
        const float *db = d + (size_t)b * os;
        const int *ib = l->idx + (size_t)b * os;
        for (int o = 0; o < os; o++) dx[ib[o]] += db[o];
    }
}

const float *mn_forward(mn_net *net, const float *x, int n)
{
    if (!net || !x || n <= 0 || ensure(net, n)) return NULL;
    int hw = net->h * net->w;
    for (int b = 0; b < n; b++)
        for (int c = 0; c < net->c; c++) {
            size_t o = ((size_t)b * net->c + c) * hw;
            float m = net->mean[c], r = 1.0f / net->std[c];
            for (int i = 0; i < hw; i++) net->in[o + i] = (x[o + i] - m) * r;
        }
    const float *a = net->in;
    for (int i = 0; i < net->count; i++) {
        layer *l = &net->layers[i];
        switch (l->kind) {
        case L_CONV: conv_forward(net, l, a, n); break;
        case L_DENSE: dense_forward(l, a, n); break;
        case L_RELU: relu_forward(l, a, n); break;
        case L_MAXPOOL: pool_forward(l, a, n); break;
        }
        a = l->out;
    }
    net->n = n;
    return a;
}

float mn_backward(mn_net *net, const int *y, int n)
{
    if (!net || !y || !net->count || n != net->n) return NAN;
    int k = mn_output_size(net);
    const float *logits = net->layers[net->count - 1].out;
    for (int b = 0; b < n; b++)
        if (y[b] < 0 || y[b] >= k) return NAN;
    mn_softmax(logits, net->dout, n, k);
    double loss = 0.0;
    for (int b = 0; b < n; b++) {
        float *g = net->dout + (size_t)b * k;
        float py = g[y[b]];
        loss -= log(py > 1e-12f ? py : 1e-12f);
        g[y[b]] -= 1.0f;
        for (int j = 0; j < k; j++) g[j] /= n;
    }
    for (int i = 0; i < net->count; i++)
        if (net->layers[i].np) memset(net->layers[i].g, 0, net->layers[i].np * sizeof(float));
    const float *d = net->dout;
    for (int i = net->count - 1; i >= 0; i--) {
        layer *l = &net->layers[i];
        const float *in = i ? net->layers[i - 1].out : net->in;
        int need = i > 0;
        switch (l->kind) {
        case L_CONV: conv_backward(net, l, in, d, n, need); break;
        case L_DENSE: dense_backward(l, in, d, n, need); break;
        case L_RELU: if (need) relu_backward(l, d, n); break;
        case L_MAXPOOL: if (need) pool_backward(l, d, n); break;
        }
        d = l->delta;
    }
    return (float)(loss / n);
}

void mn_step(mn_net *net, float lr, float wd)
{
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    net->t++;
    float c1 = 1.0f - powf(b1, (float)net->t);
    float c2 = 1.0f - powf(b2, (float)net->t);
    for (int i = 0; i < net->count; i++) {
        layer *l = &net->layers[i];
        for (int j = 0; j < l->np; j++) {
            float g = l->g[j];
            l->m[j] = b1 * l->m[j] + (1.0f - b1) * g;
            l->v[j] = b2 * l->v[j] + (1.0f - b2) * g * g;
            float u = (l->m[j] / c1) / (sqrtf(l->v[j] / c2) + eps);
            if (j < l->nw) u += wd * l->p[j];
            l->p[j] -= lr * u;
        }
    }
}

float mn_train(mn_net *net, const float *x, const int *y, int n, float lr, float wd)
{
    if (!mn_forward(net, x, n)) return NAN;
    float loss = mn_backward(net, y, n);
    if (!isnan(loss)) mn_step(net, lr, wd);
    return loss;
}

void mn_softmax(const float *logits, float *probs, int n, int k)
{
    for (int b = 0; b < n; b++) {
        const float *z = logits + (size_t)b * k;
        float *p = probs + (size_t)b * k;
        float mx = z[0];
        for (int j = 1; j < k; j++)
            if (z[j] > mx) mx = z[j];
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            p[j] = expf(z[j] - mx);
            sum += p[j];
        }
        for (int j = 0; j < k; j++) p[j] /= sum;
    }
}

int mn_save(const mn_net *net, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int32_t head[5] = {1, net->c, net->h, net->w, net->count};
    int ok = fwrite(MAGIC, 1, 4, f) == 4 && fwrite(head, sizeof head, 1, f) == 1 &&
             fwrite(net->mean, sizeof(float), net->c, f) == (size_t)net->c &&
             fwrite(net->std, sizeof(float), net->c, f) == (size_t)net->c;
    for (int i = 0; ok && i < net->count; i++) {
        const layer *l = &net->layers[i];
        int32_t a[5] = {l->kind, 0, 0, 0, 0};
        if (l->kind == L_CONV) {
            a[1] = l->oc;
            a[2] = l->size;
            a[3] = l->stride;
            a[4] = l->pad;
        } else if (l->kind == L_DENSE) {
            a[1] = l->oc;
        } else if (l->kind == L_MAXPOOL) {
            a[1] = l->size;
            a[2] = l->stride;
        }
        ok = fwrite(a, sizeof a, 1, f) == 1 &&
             (!l->np || fwrite(l->p, sizeof(float), l->np, f) == (size_t)l->np);
    }
    if (fclose(f)) ok = 0;
    return ok ? 0 : -1;
}

mn_net *mn_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    mn_net *net = NULL;
    char magic[4];
    int32_t head[5];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, MAGIC, 4) ||
        fread(head, sizeof head, 1, f) != 1 || head[0] != 1 || head[4] < 0 || head[4] > 4096)
        goto fail;
    net = mn_new(head[1], head[2], head[3]);
    if (!net) goto fail;
    if (fread(net->mean, sizeof(float), net->c, f) != (size_t)net->c ||
        fread(net->std, sizeof(float), net->c, f) != (size_t)net->c)
        goto fail;
    for (int i = 0; i < head[4]; i++) {
        int32_t a[5];
        if (fread(a, sizeof a, 1, f) != 1) goto fail;
        int r;
        switch (a[0]) {
        case L_CONV: r = mn_conv(net, a[1], a[2], a[3], a[4]); break;
        case L_DENSE: r = mn_dense(net, a[1]); break;
        case L_RELU: r = mn_relu(net); break;
        case L_MAXPOOL: r = mn_maxpool(net, a[1], a[2]); break;
        default: r = -1;
        }
        if (r) goto fail;
        layer *l = &net->layers[net->count - 1];
        if (l->np && fread(l->p, sizeof(float), l->np, f) != (size_t)l->np) goto fail;
    }
    fclose(f);
    return net;
fail:
    fclose(f);
    mn_free(net);
    return NULL;
}
