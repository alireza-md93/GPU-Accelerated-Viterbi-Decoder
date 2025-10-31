CUDA_COMPILER = nvcc
CUDA_FLAGS = -arch=sm_75 -lineinfo
CPP_COMPILER = nvcc
CPP_FLAGS = -O3 -std=c++17 -arch=sm_75

all: obj/vit_main.o obj/viterbi.o
	nvcc -o main $^

obj/%.o: src/%.cu
	$(CUDA_COMPILER) $(CPP_FLAGS) $(CUDA_FLAGS) -c $< -o $@

obj/%.o: src/%.cpp
	$(CPP_COMPILER) $(CPP_FLAGS) -c $< -o $@

clean:
	rm -f obj/*.o main
