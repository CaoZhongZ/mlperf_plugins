#pragma once

#include<iostream>

namespace intel_mlperf {

void print_int8_2Dmatrix(const int8_t* ptr, int row, int col, int stride);
void print_int32_2Dmatrix(const int* ptr, int row, int col, int stride);

void print_zero_pos_int32(const int* ptr, int row, int col, int stride);
void print_zero_pos_int8(const int8_t* ptr, int row, int col, int stride);

template<class T>
inline void print_2d_matrix(const T* ptr, int row, int col, int stride) {
    auto p = reinterpret_cast<const T (*)[stride]>(ptr);
    printf("---------------print 2d matrix------------------\n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d\t", static_cast<int>(p[i][j]));
        }
        printf("\n");
    }
    printf("---------------print 2d matrix------------------\n");
}

template<>
inline void print_2d_matrix(const uint8_t* ptr, int row, int col, int stride) {
    auto p = reinterpret_cast<const uint8_t (*)[stride]>(ptr);
    printf("---------------print 2d matrix------------------\n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d\t", static_cast<unsigned>(p[i][j]));
        }
        printf("\n");
    }
    printf("---------------print 2d matrix------------------\n");
}

}