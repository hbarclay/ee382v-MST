CXX := g++
CXX_FLAGS := -std=c++11 -Wall
LDFLAGS := 
INCLUDE := ./
BUILD := ./
OBJ_DIR := $(BUILD)/objects
SRC := $(wildcard *.cpp)
CU_SRC := $(wildcard *.cu)
OBJS := $(SRC:%.cpp=$(OBJ_DIR)/%.o) 
CU_OBJS := $(CU_SRC:%.cu=$(OBJ_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

# NVCC 
CUDA_ROOT := /usr/local/cuda-9.1

NVCC := nvcc
NVCC_FLAGS :=
CUDA_INC_DIR := -I$(CUDA_ROOT)/include
CUDA_LIB_DIR := -L$(CUDA_ROOT)/lib64
CUDA_LIBS := -lcudart

EXE := mst

all: $(EXE)

$(OBJ_DIR)/%.o: %.cpp Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD -I$(INCLUDE)

$(OBJ_DIR)/%.o : %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

seq: $(OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o seq $^ $(LDFLAGS)

$(EXE): $(CU_OBJS) $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(CU_OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LIBS)

executable: mst.cu prim_mst_gpu.cu
	nvcc -o executable mst.cu prim_mst_gpu.cu

-include $(DEPS)

.PHONY: all clean

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf executable
	-@rm -rvf seq
