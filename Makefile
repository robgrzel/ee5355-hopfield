################################################################################
#
# Build script for project
#
################################################################################

EXECUTABLES := simple_test test_driver data_driver accuracy_data_driver mincut_driver mincut_data_driver tsp queens_driver

# CUDA source files (compiled with cudacc)
CUFILES	    := evaluate_sparse.cu evaluate_dense.cu evaluate_dense_bit.cu evaluate_dense_block.cu evaluate_dense_block_coarse.cu evaluate_dense_coarse.cu evaluate_dense_cutoff.cu evaluate_sparse_ell.cu evaluate_sparse_ell_coal.cu  evaluate_sparse_jds.cu evaluate_sparse_queue.cu evaluate_sparse_warp.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES	    := hopfield.cpp evaluate_dense.cpp evaluate_sparse.cpp assoc_memory.cpp training_hebbian.cpp training_storkey.cpp mincut.cpp queens.cpp
# Header files included by any of CUFILES
CUHEADERS   := hopfield.hpp TSP_graph.hpp queens.hpp
# Header files included by any of CCFILES
CCHEADERS   := hopfield.hpp assoc_memory.hpp mincut.hpp utils.hpp TSP_graph.hpp queens.hpp

SRCDIR      := src
ROOTDIR     := .

PARENTBINDIR := bin
PARENTOBJDIR := obj

ifeq ($(dbg),1)
  ROOTBINDIR  := $(PARENTBINDIR)/debug
  OBJDIR      := $(PARENTOBJDIR)/debug
else
  ROOTBINDIR  := $(PARENTBINDIR)/release
  OBJDIR      := $(PARENTOBJDIR)/release
endif

CU_DEPS     := $(addprefix $(SRCDIR)/, $(CUHEADERS)) Makefile
C_DEPS      := $(addprefix $(SRCDIR)/, $(CCHEADERS)) Makefile

CUOBJS      := $(patsubst %.cu, $(OBJDIR)/%.cu.o, $(CUFILES))
CCOBJS      := $(patsubst %.cpp, $(OBJDIR)/%.cpp.o, $(CCFILES))
DRIVEROBJS  := $(patsubst %, $(OBJDIR)/%.cpp.o, $(EXECUTABLES))

BINS        := $(addprefix $(ROOTBINDIR)/, $(EXECUTABLES))

NVCC        := nvcc
NVCCFLAGS   += -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_61,code=sm_61 -Wno-deprecated-gpu-targets -m64 -DUNIX -std=c++11 --compiler-options -fno-strict-aliasing --compiler-options -fopenmp

CXX         := g++
CXXFLAGS    += -fopenmp -fno-strict-aliasing -m64 -std=gnu++11 -Wall -Wextra -DVERBOSE -DUNIX

LIB         += -lgomp -L/usr/local/cuda-8.0/lib64/ -lcudart -lcusparse

ifeq ($(dbg),1)
  CXXFLAGS  += -g3 -ggdb -DDEBUG
  NVCCFLAGS += -g -G -lineinfo -DDEBUG
else
  CXXFLAGS  += -O3 -DNDEBUG
  NVCCFLAGS += -O3 -DNDEBUG
endif

ifeq ($(verbose),1)
  V         := 
else
  V         := @
endif

.PHONY : all clean clobber

all : $(BINS) $(CUOBJS) $(CCOBJS) $(DRIVEROBJS) $(OBJDIR)/device.o

$(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu $(CU_DEPS) | $(OBJDIR)
	$(V)$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)/%.cpp $(C_DEPS) | $(OBJDIR)
	$(V)$(CXX) -c $(CXXFLAGS) -o $@ $<

$(OBJDIR)/device.o : $(CUOBJS) | $(OBJDIR)
	$(V)$(NVCC) $(NVCCFLAGS) -dlink $(CUOBJS) -o $@

$(ROOTBINDIR)/% : $(OBJDIR)/%.cpp.o $(CCOBJS) $(CUOBJS) $(OBJDIR)/device.o | $(ROOTBINDIR)
	$(V)$(CXX) -o $@ $^ $(LIB)

$(OBJDIR) :
	$(V)mkdir -p $(OBJDIR)

$(ROOTBINDIR) :
	$(V)mkdir -p $(ROOTBINDIR)

clean :
	$(V)rm -f $(BINS) $(CUOBJS) $(CCOBJS) $(OBJDIR)/device.o

clobber : clean
	$(V)rm -rf $(PARENTBINDIR) $(PARENTOBJDIR)
