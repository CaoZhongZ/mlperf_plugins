CXX=icpc
CXXFLAGS=-fPIC -fopenmp -xSAPPHIRERAPIDS -fma -O3 -std=c++14 -Wall -Dusercp

# CXX=g++
# CXXFLAGS=-fPIC -fopenmp -march=native -mfma -O3 -std=c++14 -Wall -Dusercp

kernels=i_gelu_tpp.cpp i_layernorm_tpp.cpp i_mha_tpp.cpp i_softmax_tpp.cpp i_gelu_tpp.hpp \
				i_layernorm_tpp.hpp i_mha_tpp.hpp  i_softmax_tpp.hpp

all : libtpps.a

libtpps.a : $(kernels:.cpp=.o)
	ar rcs $@ $^

main : main.cpp

clean :
	rm -f *.a *.o
