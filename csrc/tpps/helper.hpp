#pragma once

#include <iostream>

namespace intel_mlperf
{

void print_int8_2Dmatrix(const int8_t *ptr, int row, int col, int stride);
void print_int32_2Dmatrix(const int *ptr, int row, int col, int stride);

void print_zero_pos_int32(const int *ptr, int row, int col, int stride);
void print_zero_pos_int8(const int8_t *ptr, int row, int col, int stride);

void set_data_act(void *a, size_t n_tile);
void set_data_wei(void *w, void* b);

template <class T>
inline void print_2d_matrix(const T *ptr, int row, int col, int stride);

template <class T>
inline void compare_matrix(const T *a, const T *b, int row, int col, int lda, int ldb);

void compare_naive_input(int* a, int8_t* b, int row, int col, int lda, int ldb);
void compare_naive_weight(int* a, int8_t* b, int row, int col, int lda, int ldb);
void compare_naive_output(int* a, int8_t* b, int row, int col, int lda, int ldb);

}