CXX=icpc
OPT=-g -O2
CXXFLAGS=-fPIC -fopenmp -xHOST -fma $(OPT) -std=c++14 -Wall -Dusercp -ftz
# CXX=g++
# OPT=-O3
# CXXFLAGS=-fPIC -fopenmp -march=native -mfma $(OPT) -std=c++14 -Dusercp

kernels=i_gelu_tpp.cpp i_layernorm_tpp.cpp i_mha_tpp.cpp i_softmax_tpp.cpp i_gelu_tpp.hpp \
				i_layernorm_tpp.hpp i_mha_tpp.hpp  i_softmax_tpp.hpp

all : libtpps.a

libtpps.a : $(kernels:.cpp=.o)
	ar rcs $@ $^

mha_test : mha_test.cpp libtpps.a

main : main.cpp libtpps.a

clean :
	rm -f *.a *.o main mha_test
