#include "Matrix.h"
#include "image.h"

class ConvLayer {

    public:
        ConvLayer(int outputChannels, int stride, int filterSize, int depth, int padding, int inputWidth, int inputHeight);
        ~ConvLayer();
        Matrix* feedForward(Matrix *rawInput);
        Matrix* backPropagation(Matrix *dOut, float learningRate);
        Matrix *filters{}, *bias{}, *input{};
        int outputChannels{}, stride{}, filterSize{}, padding{}, inputChannels{}, inputWidth{}, inputHeight{};
        int colWidth{}, colHeight{}, colChannels{}, inputDim{}, outputDim{}, colInputDim{};

    private:
        void BiasAndReLU(Matrix *conv);
        void updateWeights(Matrix *dWeights, Matrix *dBias, float learningRate, int batchSize); 
};