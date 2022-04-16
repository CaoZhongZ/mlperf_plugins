# CXX=icpc
# CXXFLAGS=-fPIC -fopenmp -xHOST -fma -O3 -std=c++14 -Wall -Dusercp

CXX = g++
CXXFLAGS = -fPIC -fopenmp -march=native -mfma -O3 -std=c++14 -Wall -Dusercp

SRCS = main.cpp helper.cpp amx_config.cpp i_linear_tpp.cpp

OBJS = $(SRCS:.cpp=.o)

MAIN = xxx.exe

.PHONY: depend clean

all: $(MAIN)
	@echo Simple compiler name xxx.exe has been compiled
$(MAIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(MAIN) $(OBJS)

# libtpps.a : $(kernels:.cpp=.o)
# 	ar rcs $@ $^

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :
	$(RM) *.o *~ $(MAIN)

# depend: $(SRCS)
# 	makedepend $(INCLUDES) $^
