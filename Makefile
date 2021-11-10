CXX := g++
CXX_FLAGS := -std=c++11 -Wall
LDFLAGS := 
INCLUDE := ./
BUILD := ./
OBJ_DIR := $(BUILD)/objects
SRC := $(wildcard *.cpp)
OBJS := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

all: executable

$(OBJ_DIR)/%.o: %.cpp Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -MMD -I$(INCLUDE)

seq: $(OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o seq $^ $(LDFLAGS)


executable: mst.cu prim_mst_gpu.cu
	nvcc -o executable mst.cu prim_mst_gpu.cu

boruvka: boruvka_mst_gpu.o graph.o boruvka.cpp graph.h
	nvcc -g -arch=compute_35 -o boruvka boruvka.cpp boruvka_mst_gpu.o graph.o

graph.o: graph.cpp graph.h
	nvcc -g -c -arch=compute_35 -rdc=true graph.cpp

# temporary for checking compile errors
boruvka_mst_gpu.o: boruvka_mst_gpu.cu graph.h
	nvcc -g -c -arch=compute_35 -rdc=true boruvka_mst_gpu.cu

-include $(DEPS)

.PHONY: all clean

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf executable
	-@rm -rvf seq
