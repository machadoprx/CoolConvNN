#include "image.h"

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE

Matrix* iam2cool(Matrix *W, int channels, int width, int height, int filterSize, int stride, int pad) {

    int colWidth = ((W->rows - filterSize + (2 * pad)) / stride) + 1;
    int colHeight = ((W->columns - filterSize + (2 * pad)) / stride) + 1;
    int colChannels = channels * filterSize * filterSize;

    auto R = new Matrix(channels * filterSize * filterSize, colWidth * colHeight);
    
    for (int c = 0; c < colChannels; c++) {
        int wOffset = c % filterSize;
        int hOffset = (c / filterSize) % filterSize;
        int imageChannel = c / (filterSize * filterSize);
        for (int y = 0; y < colHeight; y++) {
            for (int x = 0; x < colWidth; x++) {
                int imageRow = (hOffset + (x * stride)) - pad;
                int imageCol = (wOffset + (y * stride)) - pad;
                int colIndex = (c * colHeight + y) * colWidth + x;
                if (imageRow < 0 || imageCol < 0 || imageRow >= height || imageCol >= width) {
                    R->data[colIndex] = 0;
                }
                else {
                    R->data[colIndex] = W->data[imageCol + width * (imageRow + height * imageChannel)];
                }
            }
        }
    }
    return R;
}

Matrix* cool2ami(Matrix *W, int channels, int width, int height, int filterSize, int stride, int pad) {

    int colWidth = ((W->rows - filterSize + (2 * pad)) / stride) + 1;
    int colHeight = ((W->columns - filterSize + (2 * pad)) / stride) + 1;
    int colChannels = channels * filterSize * filterSize;

    auto R = new Matrix(channels * filterSize * filterSize, colWidth * colHeight);
    
    for (int c = 0; c < colChannels; c++) {
        int wOffset = c % filterSize;
        int hOffset = (c / filterSize) % filterSize;
        int imageChannel = c / (filterSize * filterSize);
        for (int y = 0; y < colHeight; y++) {
            for (int x = 0; x < colWidth; x++) {
                int imageRow = (hOffset + (x * stride)) - pad;
                int imageCol = (wOffset + (y * stride)) - pad;
                int colIndex = (c * colHeight + y) * colWidth + x;
                if (imageRow < 0 || imageCol < 0 || imageRow >= height || imageCol >= width) {
                    continue;
                }
                else {
                    R->data[imageCol + width * (imageRow + height * imageChannel)] += W->data[colIndex];
                }
            }
        }
    }
    return R;
}