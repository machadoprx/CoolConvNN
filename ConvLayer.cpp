#include "ConvLayer.h"

// image format: c * height * width + y * width + x
ConvLayer::ConvLayer(int outputChannels, int stride, int filterSize, int inputChannels, int padding, int inputWidth, int inputHeight) {

    this->outputChannels = outputChannels; 
    this->stride = stride;
    this->filterSize = filterSize; 
    this->inputChannels = inputChannels; 
    this->padding = padding;
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;

    filters = new Matrix(outputChannels, filterSize * filterSize * inputChannels);
    bias = new Matrix(1, outputChannels);
}

void ConvLayer::BiasAndReLU(Matrix *conv, int size) {
    for (int i = 0; i < outputChannels; i++) {
        int index = i * size; 
        for (int j = 0; j < size; j++) {
            conv->data[index] += bias->data[i];
            conv->data[index] = Matrix::ReLU(conv->data[index]);
            index++;
        }
    }
}

void ConvLayer::fillOutput(Matrix *convolution, int offset, int size) {
    for (int i = 0; i < size; i++) {
        output->data[i + offset] = convolution->data[i];
    }
}

void ConvLayer::feedForward(Matrix *input) {

    delete output;
    
    int colWidth = ((inputWidth - filterSize + (2 * padding)) / stride) + 1;
    int colHeight = ((inputHeight - filterSize + (2 * padding)) / stride) + 1;
    int convFilterSize = colWidth * colHeight;
    int convSize = convFilterSize * outputChannels;
    int offSet = 0;

    output = new Matrix(input->rows, convSize);

    for (int i = 0; i < input->rows; i++) {
        float *img = input->data + (i * inputWidth * inputHeight * inputChannels);
        auto colInput = iam2cool(img, inputChannels, inputWidth, inputHeight, filterSize, stride, padding);
        auto convolution = filters->multiply(colInput);
        BiasAndReLU(convolution, convFilterSize);
        fillOutput(convolution, offSet, convSize);
        offSet += convSize;

        delete colInput;
        delete convolution;
    }
}

ConvLayer::~ConvLayer() {

    delete filters;

}