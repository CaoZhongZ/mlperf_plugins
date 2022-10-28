# CXX=icpc
# CXXFLAGS=-fPIC -fopenmp -xHOST -fma -O3 -std=c++14 -Wall -Dusercp

CXX = clang++
CXXFLAGS = -fPIC -fopenmp -march=native -mfma -O3 -std=c++14 -Wall -Dusercp

SRCS = main.cpp helper.cpp i_linear_tpp.cpp test_gemm.cpp ../amx_init.cpp

OBJS = $(SRCS:.cpp=.o)

MAIN = xxx.exe

.PHONY: depend clean

all: $(MAIN)
	@echo Simple compiler name xxx.exe has been compiled
$(MAIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS)

# libtpps.a : $(kernels:.cpp=.o)
# 	ar rcs $@ $^

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :
	$(RM) *.o *~ $(MAIN)

# depend: $(SRCS)
# 	makedepend $(INCLUDES) $^
