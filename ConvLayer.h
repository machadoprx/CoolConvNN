#include "Matrix.h"

class ConvLayer {

    public:
        ConvLayer(int N, int stride, int size, int depth, int padding);

    private:
        int N{}, stride{}, size{}, padding{};
        Matrix *filters{};

    

};