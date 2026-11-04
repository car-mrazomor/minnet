#ifndef MINNET_H
#define MINNET_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mn_net mn_net;

mn_net *mn_new(int c, int h, int w);
void mn_free(mn_net *net);

int mn_conv(mn_net *net, int filters, int size, int stride, int pad);
int mn_dense(mn_net *net, int outputs);
int mn_relu(mn_net *net);
int mn_maxpool(mn_net *net, int size, int stride);

void mn_init(mn_net *net, unsigned seed);
void mn_set_norm(mn_net *net, const float *mean, const float *std);

void mn_input_shape(const mn_net *net, int *c, int *h, int *w);
int mn_input_size(const mn_net *net);
int mn_output_size(const mn_net *net);
int mn_layer_count(const mn_net *net);
int mn_param_count(const mn_net *net, int layer);
float *mn_params(mn_net *net, int layer);
float *mn_grads(mn_net *net, int layer);

const float *mn_forward(mn_net *net, const float *x, int n);
float mn_backward(mn_net *net, const int *y, int n);
void mn_step(mn_net *net, float lr, float wd);
float mn_train(mn_net *net, const float *x, const int *y, int n, float lr, float wd);
void mn_softmax(const float *logits, float *probs, int n, int k);

int mn_save(const mn_net *net, const char *path);
mn_net *mn_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif
