#include "ConvLayer.h"

ConvLayer::ConvLayer(int N, int stride, int ksize, int depth, int padding) {

    this->N = N;
    this->stride = stride;
    this->ksize = ksize;
    this->padding = padding;

    filters = new Matrix(ksize * ksize * depth, N);

}
// im2col: Nx(size * size * depth)

//output shape: ((input_size - filter_size) / stride) + 1

ConvLayer::~ConvLayer() {

    delete filters;

}