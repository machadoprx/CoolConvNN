#include "png2array.h"

float *decode_png(const char* filename, int *w, int *h) {
    unsigned error;
    unsigned char* image = 0;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, filename);
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
    float *pixels = aligned_alloc(CACHE_LINE, sizeof(float) * width * height);

    for (int i = 0; i < (int)(width * height); i++) {
        int r = image[i * 4 + 0];
        int g = image[i * 4 + 1];
        int b = image[i * 4 + 2];
        float gray = (r * 0.299f) + (g * 0.587f) + (b * 0.114f);
        pixels[i] = gray;
    }
    *w = width;
    *h = height;

    free(image);
    return pixels;
}

void load_data(const char* path, float*** data, float** mean, float** var, int *labels, int *label_samples, int *in_dim) {

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        printf("error reading data file\n");
        exit(0);
    }

    fread(labels, sizeof(int), 1, fp);
    fread(label_samples, sizeof(int), 1, fp);
    fread(in_dim, sizeof(int), 1, fp);

    *mean = aligned_alloc(CACHE_LINE, sizeof(float) * (*in_dim));
    *var = aligned_alloc(CACHE_LINE, sizeof(float) * (*in_dim));
    *data = aligned_alloc(CACHE_LINE, sizeof(float*) * (*labels) * (*label_samples));

    fread(*mean, sizeof(float) * (*in_dim), 1, fp);
    fread(*var, sizeof(float) * (*in_dim), 1, fp);

    for (int i = 0; i < (*labels) * (*label_samples); i++) {
        (*data)[i] = aligned_alloc(CACHE_LINE, sizeof(float) * (*in_dim));
        fread((*data)[i], sizeof(float) * (*in_dim), 1, fp);
    }

    fclose(fp);
}

int* gen_targets(int num_labels, int label_samples) {

    int *targets = aligned_alloc(CACHE_LINE, sizeof(int) * num_labels * label_samples);

    for (int i = 0; i < num_labels; i++) {
        int label = i * label_samples;
        for (int j = 0; j < label_samples; j++) {
            targets[label + j] = i;
        }
    }

    return targets;
}

