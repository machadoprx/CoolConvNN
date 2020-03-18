#include "PoolLayer.h"
#include <iostream>

PoolLayer::PoolLayer(int stride, int filterSize, int depth, int padding, int inputWidth, int inputHeight) {
    this->stride = stride;
    this->filterSize = filterSize;  
    this->padding = padding;
    this->inputChannels = depth;
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->newWidth = 1 + (inputWidth + padding - filterSize) / stride;
    this->newHeight = 1 + (inputHeight + padding - filterSize) / stride;
    this->outputDim = newWidth * newHeight * inputChannels;
    this->inputDim = inputHeight * inputWidth * inputChannels;
}

Matrix* PoolLayer::feedForward(Matrix *rawInput) {

    delete[] indexes;

    int offSet = -padding / 2; // stride

    indexes = new int[rawInput->rows * outputDim];
    auto output = new Matrix(rawInput->rows, outputDim);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int b = 0; b < rawInput->rows; b++) {
            for (int c = 0; c < inputChannels; c++) {
                for(int h = 0; h < newHeight; h++) {
                    for(int w = 0; w < newWidth; w++) {
                        int outIndex = newWidth * ((inputChannels * b + c) * newHeight + h) + w;
                        float max = -9999999.0f;
                        int maxIndex = -1;
                        for (int fh = 0; fh < filterSize; fh++) {
                            for (int fw = 0; fw < filterSize; fw++) {
                                int currWidth = offSet + (w * stride) + fw;
                                int currHeight = offSet + (h * stride) + fh;
                                int inIndex = currWidth + inputWidth * (currHeight + inputHeight * (c + b * inputChannels));
                                if (currHeight >= 0 && currHeight < inputHeight && 
                                    currWidth >= 0  && currWidth < inputWidth) {
                                    if (rawInput->data[inIndex] > max) {
                                        maxIndex = inIndex;
                                        max = rawInput->data[inIndex];
                                    }
                                }
                            }
                        }
                        output->data[outIndex] = max;
                        indexes[outIndex] = maxIndex;
                    }
                }
            }
        }
    }
    return output;
}

Matrix* PoolLayer::backPropagation(Matrix* dOut) {

    auto R = new Matrix(dOut->rows, inputDim);
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < outputDim * dOut->rows; ++i) {
            R->data[ indexes[i] ] += dOut->data[i]; // warning if parallel and overlap
        }
    }

    return R;
}

PoolLayer::~PoolLayer() {
    delete[] indexes;
}