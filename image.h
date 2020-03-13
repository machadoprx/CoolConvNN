#ifndef NNCPP_EXEC_IMAGE_H
#define NNCPP_EXEC_IMAGE_H

#include "Matrix.h"

Matrix* iam2cool(float *im, int channels, int width, int height, int filterSize, int stride, int pad, int colWidth, int colHeight, int colChannels);
Matrix* cool2ami(float *cols, int channels, int width, int height, int filterSize, int stride, int pad, int colWidth, int colHeight, int colChannels);

#endif //NNCPP_EXEC_IMAGE_H