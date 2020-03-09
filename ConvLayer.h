#include "Matrix.h"
#include "image.h"

class ConvLayer {

    public:
        ConvLayer(int outputChannels, int stride, int filterSize, int depth, int padding, 
                    int inputWidth, int inputHeight, bool hidden);
        ~ConvLayer();
        Matrix* feedForward(Matrix *rawInput);
        Matrix* backPropagation(Matrix *dOut, float learningRate);

    private:
        int outputChannels{}, stride{}, filterSize{}, padding{}, inputChannels{}, inputWidth{}, inputHeight{};
        Matrix *filters{}, *bias{}, *input;
        bool hidden;
        void BiasAndReLU(Matrix *conv, int size);
        void fillOutput(float *convolution, int offset, int size, Matrix *output);
        void updateWeights(Matrix *dWeights, Matrix *dBias, float learningRate); 
};