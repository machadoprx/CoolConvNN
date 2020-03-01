#include "Matrix.h"
#include "image.h"

class ConvLayer {

    public:
        ConvLayer(int outputChannels, int stride, int filterSize, int depth, int padding, int inputWidth, int inputHeight);
        void feedForward(Matrix *input);
        //void backPropagation(Matrix *dOut, )

    private:
        int outputChannels{}, stride{}, filterSize{}, padding{}, inputChannels{}, inputWidth{}, inputHeight{}, oldVolume{};
        Matrix *filters{}, *bias{}, *output;

        void BiasAndReLU(Matrix *conv, int size);
        void fillOutput(Matrix *convolution, int offset, int size);
};