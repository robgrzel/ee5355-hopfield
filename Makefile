################################################################################
#
# Build script for project
#
################################################################################

EXECUTABLE  := hopfield

# CUDA source files (compiled with cudacc)
CUFILES	    := recall_dense.cu recall_sparse.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES	    := training_hebbian.cpp training_storkey.cpp recall_dense.cpp recall_sparse.cpp main.cpp
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

NVCC        := nvcc
NVCCFLAGS   += -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_61,code=sm_61 -Wno-deprecated-gpu-targets -m64 -DUNIX -std=c++11 --compiler-options -fno-strict-aliasing

CXX         := g++
CXXFLAGS    += -fopenmp -fno-strict-aliasing -m64 -std=gnu++11 -Wall -Wextra -DVERBOSE -DUNIX

LIB         += -lgomp -lcudart

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

all : $(ROOTBINDIR)/$(EXECUTABLE)

$(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu $(CU_DEPS) | $(OBJDIR)
	$(V)$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)/%.cpp $(C_DEPS) | $(OBJDIR)
	$(V)$(CXX) -c $(CXXFLAGS) -o $@ $<

$(OBJDIR)/device.o : $(CUOBJS) | $(OBJDIR)
	$(V)$(NVCC) $(NVCCFLAGS) -dlink $(CUOBJS) -o $@

$(ROOTBINDIR)/$(EXECUTABLE) : $(CUOBJS) $(CCOBJS) $(OBJDIR)/device.o | $(ROOTBINDIR)
	$(V)$(CXX) -o $@ $(CUOBJS) $(CCOBJS) $(OBJDIR)/device.o $(LIB)

$(OBJDIR) :
	$(V)mkdir -p $(OBJDIR)

$(ROOTBINDIR) :
	$(V)mkdir -p $(ROOTBINDIR)

clean :
	$(V)rm -f $(ROOTBINDIR)/$(EXECUTABLE) $(CUOBJS) $(CCOBJS) $(OBJDIR)/device.o

clobber : clean
	$(V)rm -rf $(PARENTBINDIR) $(PARENTOBJDIR)
