#pragma once

#include <cstdint>

enum Metric {B16=16, B32=32};
enum Input {HARD, SOFT4, SOFT8};

template<Metric metricType>
struct packType_helper;

template<> struct packType_helper<Metric::B16> { using type = uint16_t; };
template<> struct packType_helper<Metric::B32> { using type = uint32_t; };

template<Metric metricType>
using pack_t = typename packType_helper<metricType>::type;

template<Metric metricType>
struct metricType_helper;

template<> struct metricType_helper<Metric::B16> { using type = int16_t; };
template<> struct metricType_helper<Metric::B32> { using type = int32_t; };

template<Metric metricType>
using metric = typename metricType_helper<metricType>::type;

template<Metric metricType>
class ViterbiCUDA{
	
	public:
	
	static constexpr int constLen = 7; //constraint length
	static constexpr int polyn1 = 0171; //polynomial 1
	static constexpr int polyn2 = 0133; //polynomial 2

	static constexpr int roundup(int a, int b) { if(a <= 0) return 0; else return (((a-1)/b + 1) * b); };
	static constexpr int extraL_raw = 32;
	static constexpr int extraR_raw = 32;
	static constexpr int extraL = roundup(extraL_raw, metricType) - (constLen - 1);
	static constexpr int extraR = roundup(extraR_raw, metricType) + (constLen - 1);
	static constexpr int slideSize = 128;
	static constexpr int shMemWidth = extraL + slideSize + extraR;
    static constexpr int blockDimY = 2;
	
	ViterbiCUDA();
	ViterbiCUDA(size_t inputNum);
	~ViterbiCUDA();
	
	void run(float* input_h, pack_t<metricType>* output_h, size_t messageLen, float* kernelTime = nullptr);

	size_t getInputSize(size_t inputNum);
	size_t getMessageLen(size_t inputNum);
	size_t getOutputSize(size_t inputNum);

private:
	pack_t<metricType>* pathPrev_d;
	pack_t<metricType>* dec_d;
	float* enc_d;
	bool preAllocated;
	int blocksNum_total;


	void memAlloc(size_t inputNum);
	void memFree();
	void deviceSetup();

	size_t getSharedMemSize();
	size_t getPathPrevSize();
};