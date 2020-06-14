#include "utils.h"

void get_gauss(float mean, float stddev, float *u, float *v) {
    float x, y, s = 0.0f;
    while (s >= 1.0f || s == 0.0f) {
        x = ((float)rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        y = ((float)rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        s = x * x + y * y;
    }
    s = sqrtf(-2.0f * logf(s) / s);
    *u = mean + stddev * x * s;
    *v = mean + stddev * y * s;
}