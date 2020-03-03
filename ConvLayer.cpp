#include "ConvLayer.h"

// image format: c * height * width + y * width + x
ConvLayer::ConvLayer(int inputChannels, int outputChannels, int stride, int filterSize, int padding, 
                    int inputWidth, int inputHeight, bool hidden, bool isFirst) {

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
    int convFilterSize = colWidth * colHeight;
    int convSize = convFilterSize * outputChannels;
    int offSet = 0;

    auto output = new Matrix(input->rows, convSize);

    for (int i = 0; i < input->rows; i++) {
        float *img = input->data + (i * inputWidth * inputHeight * inputChannels);
        auto colInput = iam2cool(img, inputChannels, inputWidth, inputHeight, filterSize, stride, padding);
        auto convolution = filters->multiply(colInput);
        BiasAndReLU(convolution, convFilterSize);
        fillOutput(convolution->data, offSet, convSize, output);
        offSet += convSize;

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
                filters->data[index] -= dWeights->data[index];
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
        auto biasDerivative = dOutRow->sumRows();
        
        //update weights

        delete dWeightsRow;
        delete colInput;
        delete colInputT;
        delete biasDerivative;

        auto filtersT = filters->transposed();
        auto dInput = filtersT->multiply(dOutRow);
        auto dInputReLU = dInput->ReLUDerivative(input);
        dInputBatch->setRow(dInputReLU->data, inputDim, i);

        delete filtersT;
        delete dInput;
        delete dInputReLU;
        delete dOutRow;
    }

    return dInputBatch;
}

ConvLayer::~ConvLayer() {

    delete filters;

}