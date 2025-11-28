CUDA_COMPILER = nvcc
CUDA_FLAGS = -arch=sm_75 -lineinfo
CPP_COMPILER = nvcc
CPP_FLAGS = -O3 -std=c++17 -I./src/viterbi -I./src/dataflow
DEVICE_OBJ = obj/viterbi.o 
HOST_OBJ = obj/main.o

all: $(HOST_OBJ) $(DEVICE_OBJ)
	nvcc $(CUDA_FLAGS) -o main $^

obj/%.o: src/viterbi/%.cu
	$(CUDA_COMPILER) $(CPP_FLAGS) $(CUDA_FLAGS) -dc $< -o $@

obj/%.o: src/%.cpp
	$(CPP_COMPILER) $(CPP_FLAGS) -dc $< -o $@

clean:
	rm -f obj/*.o main
