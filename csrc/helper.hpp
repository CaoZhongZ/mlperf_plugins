#pragma once

#include<iostream>

namespace intel_mlperf {

void print_int8_2Dmatrix(const int8_t* ptr, int row, int col, int stride);
void print_int32_2Dmatrix(const int* ptr, int row, int col, int stride);

}