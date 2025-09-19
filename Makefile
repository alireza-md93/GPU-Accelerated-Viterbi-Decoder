all: obj/vit_main.o obj/viterbi.o
	nvcc -arch=sm_75 -o main $^

obj/%.o: src/%.cu
	nvcc -lineinfo -arch=sm_75 -c $< -o $@

clean:
	rm -f obj/*.o main
