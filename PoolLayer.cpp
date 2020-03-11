#include "PoolLayer.h"
#include <iostream>

PoolLayer::PoolLayer(int stride, int filterSize, int depth, int padding, int inputWidth, int inputHeight) {

    this->stride = stride;
    this->filterSize = filterSize;  
    this->padding = padding;
    this->inputChannels = depth;
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
}

void PoolLayer::fillOutput(float *input, int offset, int size, Matrix *output) {
    for (int i = 0; i < size; i++) {
        output->data[i + offset] = input[i];
    }
}

Matrix* PoolLayer::feedForward(Matrix *rawInput) {

    delete[] indexes;

    //int inputDim = inputWidth * inputHeight * inputChannels;
    int newWidth = 1 + (inputWidth + padding - filterSize) / stride;
    int newHeight = 1 + (inputHeight + padding - filterSize) / stride;
    int outputDim = newWidth * newHeight * inputChannels;
    int offSet = -padding / 2; // stride

    auto indexes = new int[rawInput->rows * outputDim];
    auto output = new Matrix(rawInput->rows, outputDim);

    for (int b = 0; b < rawInput->rows; b++) {
        for (int c = 0; c < inputChannels; c++) {
            for(int h = 0; h < newHeight; h++) {
                for(int w = 0; w < newWidth; w++) {
                    int outIndex = newWidth * ((inputChannels * b + c) * newHeight + h) + w;
                    float max = -9999999;
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
    return output;
}

Matrix* PoolLayer::backPropagation(Matrix* dOut) {
    int newWidth = 1 + (inputWidth + padding - filterSize) / stride;
    int newHeight = 1 + (inputHeight + padding - filterSize) / stride;
    int outDim = newWidth * newHeight * inputChannels;
    auto R = new Matrix(dOut->rows, outDim);
    
    for (int i = 0; i < outDim * dOut->rows; ++i) {
        R->data[ indexes[i] ] += dOut->data[i];
    }

    return R;
}

PoolLayer::~PoolLayer() {
    delete[] indexes;
}