#include "utils.h"

void get_gauss(float mean, float stddev, float *u, float *v) {
    float x = 0, y = 0, s = 0.0f;
    while (s >= 1.0f || s == 0.0f) {
        x = ((float)rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        y = ((float)rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        s = x * x + y * y;
    }
    s = sqrtf(-2.0f * logf(s) / s);
    *u = mean + stddev * x * s;
    *v = mean + stddev * y * s;
}

int dec_activation(const char* name) {
    if (strcmp(name, "none") == 0) {
        return NONE;
    }
    else if (strcmp(name, "relu") == 0) {
        return RELU;
    }
    return -1;
}

int dec_layer_type(const char* name) {
    if (strcmp(name, "fc") == 0) {
        return FC;
    }
    else if (strcmp(name, "conv") == 0) {
        return CONV;
    }
    else if (strcmp(name, "max_pool") == 0) {
        return MAX_POOL;
    }
    else if (strcmp(name, "activate") == 0) {
        return ACTIVATION;
    }
    else if (strcmp(name, "batch_norm") == 0) {
        return BN;
    }
    return -1;
}