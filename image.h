#ifndef NNCPP_EXEC_IMAGE_H
#define NNCPP_EXEC_IMAGE_H

#include "Matrix.h"

Matrix* iam2cool(float *im, int channels, int width, int height, int filterSize, int stride, int pad);
Matrix* cool2ami(float *cols, int channels, int width, int height, int filterSize, int stride, int pad);

#endif //NNCPP_EXEC_IMAGE_H