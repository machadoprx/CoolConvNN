#ifndef image_h
#define image_h

#include "matrix.h"

void iam2cool(float *im, int in_c, int in_w, int in_h, int f_size, int stride, int padd, int col_w, int col_h, int col_c, float *out);
void cool2ami(float *cols, int in_c, int in_w, int in_h, int f_size, int stride, int padd, int col_w, int col_h, int col_c, float *out);

#endif