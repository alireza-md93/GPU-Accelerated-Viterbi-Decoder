#pragma once

#include <cstdint>
#include <type_traits>

enum Metric {B16=16, B32=32};
enum Input {HARD, SOFT4, SOFT8};

template<Metric metricType>
class ViterbiCUDA{
	public:

	using metric_t = std::conditional_t<metricType == Metric::B16, int16_t, int32_t>;
	using decPack_t = std::conditional_t<metricType == Metric::B16, uint16_t, uint32_t>; // packed decoded bits (16 bits per uint16_t, 32 bits per uint32_t)
	
	static constexpr int constLen = 7; //constraint length
	static constexpr int polyn1 = 0171; //polynomial 1
	static constexpr int polyn2 = 0133; //polynomial 2

	static constexpr int roundup(int a, int b) { if(a <= 0) return 0; else return (((a-1)/b + 1) * b); };
	static constexpr int bitsPerPack = (metricType == Metric::B32) ? 32 : 16;
	static constexpr int extraL_raw = 32;
	static constexpr int extraR_raw = 32;
	static constexpr int extraL = roundup(extraL_raw, bitsPerPack) - (constLen - 1);
	static constexpr int extraR = roundup(extraR_raw, bitsPerPack) + (constLen - 1);
	static constexpr int slideSize = 128;
	static constexpr int shMemWidth = extraL + slideSize + extraR;
    static constexpr int blockDimY = 2;
	
	ViterbiCUDA();
	ViterbiCUDA(size_t inputNum);
	~ViterbiCUDA();
	
	void run(float* input_h, decPack_t* output_h, size_t messageLen, float* kernelTime = nullptr);

	size_t getInputSize(size_t inputNum);
	size_t getMessageLen(size_t inputNum);
	size_t getOutputSize(size_t inputNum);

private:
	decPack_t* pathPrev_d;
	decPack_t* dec_d;
	float* enc_d;
	bool preAllocated;
	int blocksNum_total;
	struct Impl;
	Impl* pImpl;

	void memAlloc(size_t inputNum);
	void memFree();
	void deviceSetup();

	void timerSetup();
	void timerDelete();
	void timerStart();
	void timerStop();
	float timerElapsed();

	size_t getSharedMemSize();
	size_t getPathPrevSize();
};