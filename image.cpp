#include "image.h"

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE

Matrix* iam2cool(float *im, int channels, int width, int height, int filterSize, int stride, int pad, int colWidth, int colHeight, int colChannels) {

    auto R = new Matrix(colChannels, colWidth * colHeight);
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int c = 0; c < colChannels; c++) {
            int wOffset = c % filterSize;
            int hOffset = (c / filterSize) % filterSize;
            int imageChannel = c / (filterSize * filterSize);
            for (int y = 0; y < colHeight; y++) {
                for (int x = 0; x < colWidth; x++) {
                    int imageRow = (hOffset + (y * stride)) - pad;
                    int imageCol = (wOffset + (x * stride)) - pad;
                    int colIndex = (c * colHeight + y) * colWidth + x;
                    if (imageRow < 0 || imageCol < 0 || imageRow >= height || imageCol >= width) {
                        R->data[colIndex] = 0;
                    }
                    else {
                        int imageIndex = imageCol + width * (imageRow + height * imageChannel);
                        R->data[colIndex] = im[imageIndex];
                    }
                }
            }
        }
    }

    return R;
}

Matrix* cool2ami(float *cols, int channels, int width, int height, int filterSize, int stride, int pad, int colWidth, int colHeight, int colChannels) {

    auto R = new Matrix(channels, width * height);
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int c = 0; c < colChannels; c++) {
            int wOffset = c % filterSize;
            int hOffset = (c / filterSize) % filterSize;
            int imageChannel = c / (filterSize * filterSize);
            for (int y = 0; y < colHeight; y++) {
                for (int x = 0; x < colWidth; x++) {
                    int imageRow = (hOffset + (x * stride)) - pad;
                    int imageCol = (wOffset + (y * stride)) - pad;
                    int colIndex = (c * colHeight + y) * colWidth + x;
                    int imageIndex = imageCol + width * (imageRow + height * imageChannel);
                    if (imageRow < 0 || imageCol < 0 || imageRow >= height || imageCol >= width) {
                        continue;
                    }
                    else {
                        R->data[imageIndex] += cols[colIndex];
                    }
                }
            }
        }
    }
    return R;
}