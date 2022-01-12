#include "helper.hpp"

namespace intel_mlperf {

void print_int8_2Dmatrix(const int8_t* ptr, int row, int col, int stride)
{
    printf("---------------print int8 2D matrix------------------\n");
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d\t", static_cast<int>(ptr[i*stride+j]));
        }
        printf("\n");
    }
    printf("---------------print int8 2D matrix------------------\n");
}

void print_int32_2Dmatrix(const int* ptr, int row, int col, int stride)
{
    printf("---------------print int32 2D matrix------------------\n");
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d\t", ptr[i*stride+j]);
        }
        printf("\n");
    }
    printf("---------------print int32 2D matrix------------------\n");
}

}