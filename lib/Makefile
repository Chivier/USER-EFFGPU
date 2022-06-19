NVCC=/opt/cuda/11.0
CUDA_HOME=/opt/cuda/11.0

CUDA_ARCH=compute_80
CUDA_CODE=sm_80

NVCC=$(CUDA_HOME)/bin/nvcc
MPICXX=mpicxx
CXX=$(MPICXX)

CXXFLAGS= -g -O3 -std=c++11 -DFIX_EPH_GPU
NVCCFLAGS= -g -O3 -ccbin=$(CXX) --std=c++11 -arch=$(CUDA_ARCH) -code $(CUDA_CODE) --use_fast_math -lcublas -lcublas_device -lcudadevrt -Wno-deprecated-gpu-targets

INCFLAGS= -I$(CUDA_HOME)/include -I../
LDFLAGS= -L$(CUDA_HOME)/lib64

all: eff_gpu.cu eff_gpu.cpp eff_gpu.h
	$(NVCC) -c eff_gpu.cu  -o eff_gpu_cu.o $(INCFLAGS) $(NVCCFLAGS)
	$(NVCC) -c eff_gpu.cpp -o eff_gpu_cpp.o $(INCFLAGS) $(NVCCFLAGS)
	ar rcs libeff_gpu.a eff_gpu_cu.o eff_gpu_cpp.o
	$(CXX) $(CXXFLAGS) main.cpp libeff_gpu.a -o main -lcuda -lcudart $(INCFLAGS) $(LDFLAGS)

clean: 
	rm -f libeff_gpu.a eff_gpu_cu.o eff_gpu_cpp.o main
