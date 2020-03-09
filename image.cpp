#include "image.h"

/*Matrix* concatenateBatch(Matrix **im2col, int batchSize, int colWidth, int colHeight, int colChannels) {

    int concIndex = 0;
    int columns = colWidth * colHeight;
    int rows = colChannels;

    Matrix *conc = new Matrix(rows, columns * batchSize);

    for (int i = 0; i < rows; i++) {
        for (int n = 0; n < batchSize; n++) {
            int index = i * columns;
            for (int j = 0; j < columns; j++) {
                conc->data[concIndex++] = im2col[n]->data[index++];
            }
        }
    }

    return conc;
}

Matrix** splitBatch(Matrix *conv, int batchSize, int colWidth, int colHeight, int filtersN) {

    // output form same as image:
    // image[i] = Matrix(3, width * height)
    // out[i] = Matrix(filtersN, colW * colH)

    Matrix **splitted = new Matrix*[batchSize];
    for (int i = 0; i < batchSize; i++) {
        splitted[i] = new Matrix(filtersN, colWidth * colHeight);
    }
    // Matrix(channels * filterSize * filterSize, colWidth * colHeight);

    for (int i = 0; i < rows; i++) {
        for (int n = 0; n < batchSize; n++) {
            for (int j = 0; j < columns; j++) {
                conc->data[concIndex++] = conv[n]->data[i * columns + j];
            }
        }
    }

    return conv;
}*/

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE

Matrix* iam2cool(float *im, int channels, int width, int height, int filterSize, int stride, int pad) {

    int colWidth = ((width - filterSize + (2 * pad)) / stride) + 1;
    int colHeight = ((height - filterSize + (2 * pad)) / stride) + 1;
    int fSize2 = filterSize * filterSize;
    int colChannels = channels * fSize2;

    auto R = new Matrix(colChannels, colWidth * colHeight);
    
    for (int c = 0; c < colChannels; c++) {
        int wOffset = c % filterSize;
        int hOffset = (c / filterSize) % filterSize;
        int imageChannel = c / fSize2;
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
    return R;
}

Matrix* cool2ami(float *cols, int channels, int width, int height, int filterSize, int stride, int pad) {

    int colWidth = ((width - filterSize + (2 * pad)) / stride) + 1;
    int colHeight = ((height - filterSize + (2 * pad)) / stride) + 1;
    int colChannels = channels * filterSize * filterSize;

    auto R = new Matrix(channels, width * height);
    
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
    return R;
}