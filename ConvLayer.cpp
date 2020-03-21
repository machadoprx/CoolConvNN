#include "ConvLayer.h"

ConvLayer::ConvLayer(int inputChannels, int outputChannels, int stride, int filterSize, int padding, int inputWidth, int inputHeight) {

    this->outputChannels = outputChannels; 
    this->stride = stride;
    this->filterSize = filterSize; 
    this->inputChannels = inputChannels; 
    this->padding = padding;
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->colWidth = ((inputWidth - filterSize + (2 * padding)) / stride) + 1;
    this->colHeight = ((inputHeight - filterSize + (2 * padding)) / stride) + 1;
    this->inputDim = inputWidth * inputHeight * inputChannels;
    this->outputDim = colWidth * colHeight * outputChannels;
    this->colChannels = inputChannels * filterSize * filterSize;
    this->colInputDim = colChannels * colWidth * colHeight;

    filters = new Matrix(outputChannels, colChannels);
    bias = new Matrix(1, outputChannels);
    filters->randomize(0.0f, sqrtf(2.0f / (float)outputChannels));
}

void ConvLayer::BiasAndReLU(Matrix *conv) {
    
    int size = colWidth * colHeight;

    #pragma omp parallel
    {
        #pragma omp for collapse(2) nowait
        for (int b = 0; b < conv->rows; b++) {
            for (int i = 0; i < outputChannels; i++) {
                int index = size * (b * outputChannels + i);
                for (int j = 0; j < size; j++) {
                    conv->data[index] += bias->data[i];
                    conv->data[index] = (conv->data[index] > .0f) ? conv->data[index] : .0f;
                    index++;
                }
            }
        }
    }
}

void ConvLayer::updateWeights(Matrix *dWeights, Matrix *dBias, float learningRate, int batchSize) {
    
    float invBatch = 1.0f / (float)batchSize;

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < outputChannels; i++) {
            int index = i * colChannels;
            for (int j = 0; j < colChannels; j++) {
                filters->data[index] -= (dWeights->data[index] * invBatch) * learningRate;
                index++;
            }
            bias->data[i] -= (dBias->data[i] * invBatch) * learningRate;
        }
    }
}

Matrix* ConvLayer::feedForward(Matrix *rawInput) {

    delete input;

    input = new Matrix(rawInput->rows, colInputDim);
    auto output = new Matrix(rawInput->rows, outputDim);

    for (int i = 0; i < rawInput->rows; i++) {
        auto colInput = iam2cool(rawInput->data + (i * inputDim), inputChannels, inputWidth, inputHeight, filterSize, stride, padding, colWidth, colHeight, colChannels);
        Matrix::mcopy(input->data + (i * colInputDim), colInput->data, colInputDim);
        auto convolution = filters->multiply(colInput);
        Matrix::mcopy(output->data + (i * outputDim), convolution->data, outputDim);

        delete colInput;
        delete convolution;
    }

    BiasAndReLU(output);

    return output;
}

Matrix* ConvLayer::backPropagation(Matrix* dOut, float learningRate) {

    assert(dOut->rows == input->rows);

    auto dInputBatch = new Matrix(input->rows, inputDim);
    auto dWeights = new Matrix(filters->rows, filters->columns);
    auto dBias = new Matrix(bias->rows, bias->columns);

    for (int i = 0; i < input->rows; i++) {
        auto colInput = new Matrix(colChannels, colWidth * colHeight);
        Matrix::mcopy(colInput->data, input->data + (i * colInputDim), colInputDim);
        auto colInputT = colInput->transposed();

        auto dOutRow = new Matrix(outputChannels, colWidth * colHeight); // fill

        Matrix::mcopy(dOutRow->data, dOut->data + (i * outputDim), outputDim);

        auto dWeightsRow = dOutRow->multiply(colInputT);
        auto dBiasRow = dOutRow->sumColumns();

        dWeights->accumulate(dWeightsRow);
        dBias->accumulate(dBiasRow);

        delete dWeightsRow;
        delete colInputT;
        delete dBiasRow;

        auto filtersT = filters->transposed(); // add filtersT and colInputT to cache
        auto dInput = filtersT->multiply(dOutRow);
        dInput->apply_reluderivative(colInput);
        auto dInputImage = cool2ami(dInput->data, inputChannels, inputWidth, inputHeight, filterSize, stride, padding, colWidth, colHeight, colChannels);
        dInputBatch->setRow(dInputImage->data, i);

        delete colInput;
        delete dInputImage;
        delete filtersT;
        delete dInput;
        delete dOutRow;
    }

    updateWeights(dWeights, dBias, learningRate, input->rows);
    delete dWeights;
    delete dBias;

    return dInputBatch;
}

ConvLayer::~ConvLayer() {
    delete filters;
    delete bias;
    delete input;
}