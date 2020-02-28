#include "Matrix.h"

class ConvLayer {

    public:
        ConvLayer(int filters, int stride, int ksize, int depth, int padding);

    private:
        int N{}, stride{}, ksize{}, padding{}, depth{};
        Matrix *filters{};

};