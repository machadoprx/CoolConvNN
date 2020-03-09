#include "ConvLayer.h"
#include <iostream>

ConvLayer::ConvLayer(int inputChannels, int outputChannels, int stride, int filterSize, int padding, 
                    int inputWidth, int inputHeight, bool hidden) {

    this->outputChannels = outputChannels; 
    this->stride = stride;
    this->filterSize = filterSize; 
    this->inputChannels = inputChannels; 
    this->padding = padding;
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;

    filters = new Matrix(outputChannels, filterSize * filterSize * inputChannels);
    bias = new Matrix(1, outputChannels);
    filters->randomize();
}

void ConvLayer::BiasAndReLU(Matrix *conv, int size) {
    for (int i = 0; i < outputChannels; i++) {
        int index = i * size; 
        for (int j = 0; j < size; j++) {
            conv->data[index] = Matrix::ReLU(conv->data[index] + bias->data[i]);
            index++;
        }
    }
}

void ConvLayer::fillOutput(float *convolution, int offset, int size, Matrix *output) {
    for (int i = 0; i < size; i++) {
        output->data[i + offset] = convolution[i];
    }
}

Matrix* ConvLayer::feedForward(Matrix *rawInput) {

    delete input;
    input = rawInput->copy();

    int colWidth = ((inputWidth - filterSize + (2 * padding)) / stride) + 1;
    int colHeight = ((inputHeight - filterSize + (2 * padding)) / stride) + 1;
    int inputDim = inputWidth * inputHeight * inputChannels;
    int convFilterSize = colWidth * colHeight;
    int convSize = convFilterSize * outputChannels;

    auto output = new Matrix(input->rows, convSize);

    for (int i = 0; i < input->rows; i++) {
        float *img = input->data + (i * inputDim);
        auto colInput = iam2cool(img, inputChannels, inputWidth, inputHeight, filterSize, stride, padding);
        auto convolution = filters->multiply(colInput);
        if (hidden) {
            BiasAndReLU(convolution, convFilterSize);
        }
        fillOutput(convolution->data, i * convSize, convSize, output);

        delete colInput;
        delete convolution;
    }

    return output;
}

void ConvLayer::updateWeights(Matrix *dWeights, Matrix *dBias, float learningRate) {
    
    int len = filterSize * filterSize * inputChannels;

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for nowait
        for (int i = 0; i < outputChannels; i++) {
            int index = i * len;
            for (int i = 0; i < len; i++) {
                filters->data[index] -= dWeights->data[index] * learningRate;
                index++;
            }
            bias->data[i] -= dBias->data[i] * learningRate;
        }

    }
}

Matrix* ConvLayer::backPropagation(Matrix* dOut, float learningRate) {

    int colWidth = ((inputWidth - filterSize + (2 * padding)) / stride) + 1;
    int colHeight = ((inputHeight - filterSize + (2 * padding)) / stride) + 1;
    int inputDim = inputWidth * inputHeight * inputChannels;
    int outputDim = outputChannels * colWidth * colHeight;

    auto dInputBatch = new Matrix(input->rows, inputDim);

    for (int i = 0; i < input->rows; i++) {

        float *inputData = input->data + (i * inputDim);

        auto dOutRow = new Matrix(outputChannels, colWidth * colHeight); // fill
        fillOutput(dOut->data, i * outputDim, outputDim, dOutRow);

        auto colInput = iam2cool(inputData, inputChannels, inputWidth, inputHeight, filterSize, stride, padding);
        auto colInputT = colInput->transposed();

        auto dWeightsRow = dOutRow->multiply(colInputT);
        auto dBiasRow = dOutRow->sumRows();
        
        //update weights
        updateWeights(dWeightsRow, dBiasRow, learningRate);

        delete dWeightsRow;
        delete colInput;
        delete colInputT;
        delete dBiasRow;

        auto filtersT = filters->transposed();
        auto dInput = filtersT->multiply(dOutRow);
        auto dInputReLU = dInput->ReLUDerivative(input);
        auto dInputImage = cool2ami(dInputReLU->data, inputChannels, inputWidth, inputHeight, filterSize, stride, padding);
        dInputBatch->setRow(dInputImage->data, inputDim, i);

        delete dInputImage;
        delete filtersT;
        delete dInput;
        delete dInputReLU;
        delete dOutRow;
    }

    return dInputBatch;
}

ConvLayer::~ConvLayer() {
    delete filters;
    delete bias;
    delete input;
}

/*
int main(int argc, char *argv[]) {
    
    float f[] = {1, 1, -1,
                 0, 0, 0,
                 0, 1, -1,
                 0, -1, 1,
                 -1, 0, 0,
                 1, 1, 1,
                 1, -1, 0,
                 1, 0, 1,
                 1, 0, 1,
                 0, 1, 1,
                 1, 0, 0,
                 -1, 0, -1,
                 0, 1, 0,
                 1, 0, 1,
                 1, 0, 0,
                 1, 1, -1,
                 0, 1, 1,
                 -1, 1, 0};
    
    for (int i = 0; i < outputChannels * filterSize * filterSize * inputChannels; i++) {
        filters->data[i] = f[i];
    }
    bias->data[0] = 1;
    bias->data[1] = 0;
    
    
    


    int inputChannels = 3;
    int width = 5, height = 5;
    int padding = 1;
    int stride = 2;
    int size = 3;
    int n = 2;

    float data[] = {1,0,2,1,1,
                    1,2,0,2,2,
                    2,1,0,2,1,
                    0,1,0,1,1,
                    1,1,0,2,0,
                    1,2,1,0,0,
                    2,1,2,2,2,
                    0,1,2,0,2,
                    2,1,2,0,1,
                    1,1,1,1,2,
                    1,0,1,0,2,
                    1,1,1,1,1,
                    2,1,1,0,0,
                    2,0,0,1,2,
                    0,1,0,2,2};
    
    ConvLayer *l = new ConvLayer(inputChannels, n, stride, size, padding, width, height, true, true);
    Matrix *input = new Matrix(1, width * height * inputChannels);
    //memcpy(input->data, data, sizeof(float) * width * height * inputChannels);
    for (int i = 0; i < width * height * inputChannels; i++) {
        input->data[i] = data[i];
    }
    Matrix *test = l->feedForward(input);

    int colWidth = ((width - size + (2 * padding)) / stride) + 1;
    int colHeight = ((height - size + (2 * padding)) / stride) + 1;

    for (int c = 0; c < n; c++) {
        for (int i = 0; i < colHeight; i++) {
            for (int j = 0; j < colWidth; j++) {
                std::cout << test->data[(c * colHeight + i) * colWidth + j] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    return 0;
}*/