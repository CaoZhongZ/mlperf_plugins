#include "helper.hpp"

namespace intel_mlperf {

void print_int8_2Dmatrix(const int8_t* ptr, int row, int col, int stride)
{
    auto p = reinterpret_cast<const int8_t (*)[stride]>(ptr);
    printf("---------------print int8 2D matrix------------------\n");
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d\t", static_cast<int>(p[i][j]));
        }
        printf("\n");
    }
    printf("---------------print int8 2D matrix------------------\n");
}

void print_int32_2Dmatrix(const int* ptr, int row, int col, int stride)
{
    auto p = reinterpret_cast<const int (*)[stride]>(ptr);
    printf("---------------print int32 2D matrix------------------\n");
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%d\t", p[i][j]);
        }
        printf("\n");
    }
    printf("---------------print int32 2D matrix------------------\n");
}

void print_zero_pos_int32(const int* ptr, int row, int col, int stride)
{
    printf("---------------print int32 zero pos------------------\n");
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if (static_cast<int>(ptr[i*stride+j]) == 0) {
                printf("[%d, %d]\n", i, j);
            }
        }
    }
    printf("---------------print int32 zero pos end------------------\n");
}

void print_zero_pos_int8(const int8_t* ptr, int row, int col, int stride)
{
    printf("---------------print int8 zero pos------------------\n");
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            if (ptr[i*stride+j] == 0) {
                printf("[%d, %d]\n", i, j);
            }
        }
    }
    printf("---------------print int8 zero pos end------------------\n");
}

}