################################################################################
#
# Build script for project
#
################################################################################

EXECUTABLES := simple_test test_driver

# CUDA source files (compiled with cudacc)
CUFILES	    := recall_dense.cu recall_sparse.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES	    := hopfield.cpp training_hebbian.cpp training_storkey.cpp recall_dense.cpp recall_sparse.cpp
# Header files included by any of CUFILES
CUHEADERS   := hopfield.hpp
# Header files included by any of CCFILES
CCHEADERS   := hopfield.hpp

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
NVCCFLAGS   += -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_61,code=sm_61 -Wno-deprecated-gpu-targets -m64 -DUNIX -std=c++11 --compiler-options -fno-strict-aliasing

CXX         := g++
CXXFLAGS    += -fopenmp -fno-strict-aliasing -m64 -std=gnu++11 -Wall -Wextra -DVERBOSE -DUNIX

LIB         += -lgomp -lcudart -lcusparse

ifeq ($(dbg),1)
  CXXFLAGS  += -g3 -ggdb
  NVCCFLAGS += -g -G -lineinfo
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
