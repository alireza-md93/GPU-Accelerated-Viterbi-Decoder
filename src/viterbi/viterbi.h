#pragma once

#include <cstdint>
#include <type_traits>
#include <cuda_fp16.h>

enum Metric {B16, B32, FP16};
enum ChannelIn {HARD, SOFT4, SOFT8, SOFT16, FP32};

template<Metric metricType, ChannelIn inputType>
class ViterbiCUDA{
	public:

	using metric_t = 	std::conditional_t<metricType == Metric::B16, int16_t, 
						std::conditional_t<metricType == Metric::B32, int32_t,
						std::conditional_t<metricType == Metric::FP16, __half, float>>>;
	using decPack_t = 	std::conditional_t<metricType == Metric::B16, uint16_t, 
						std::conditional_t<metricType == Metric::B32, uint32_t,
						std::conditional_t<metricType == Metric::FP16,uint16_t, uint32_t>>>; // packed decoded bits (16 bits per uint16_t, 32 bits per uint32_t)
	using encPack_t = std::conditional_t<inputType == ChannelIn::FP32, float, int32_t>; // packed encoded input
	
	static constexpr int constLen = 7; //constraint length
	static constexpr int polyn1 = 0171; //polynomial 1
	static constexpr int polyn2 = 0133; //polynomial 2

	static constexpr int roundup(int a, int b) { if(a <= 0) return 0; else return ((a + b - 1) / b * b); };
	static constexpr size_t roundup(size_t a, size_t b) { if(a <= 0) return 0; else return ((a + b - 1) / b * b); };
	static constexpr int bitsPerMetric = 	(metricType == Metric::B16) ? 16 : 
											(metricType == Metric::B32) ? 32 : 11;
	static constexpr int bitsPerPack = 		(metricType == Metric::B16) ? 16 : 
											(metricType == Metric::B32) ? 32 : 16;
	static constexpr int extraL_raw = 32;
	static constexpr int extraR_raw = 32;
	static constexpr int slideSize_raw = 32;
	static constexpr int extraL = roundup(extraL_raw, bitsPerPack) - (constLen - 1);
	static constexpr int extraR = roundup(extraR_raw, bitsPerPack) + (constLen - 1);
	static constexpr int slideSize = roundup(slideSize_raw, bitsPerPack);
	static constexpr int shMemWidth = extraL + slideSize + extraR;
    static constexpr int blockDimY = 2;
    static constexpr int FPprecision = 4;
	static constexpr int encDataPerPack = (inputType == ChannelIn::HARD) ? (sizeof(encPack_t) * 8) :
										(inputType == ChannelIn::SOFT4) ? (sizeof(encPack_t) * 2) :
										(inputType == ChannelIn::SOFT8) ? (sizeof(encPack_t)) :
										(inputType == ChannelIn::SOFT16) ? (sizeof(encPack_t) / 2) : 1;
	static constexpr int encDataWidth = (inputType == ChannelIn::HARD) ? 1 :
										(inputType == ChannelIn::SOFT4) ? 4 :
										(inputType == ChannelIn::SOFT8) ? 8 :
										(inputType == ChannelIn::SOFT16) ? 16 : FPprecision;
	
	ViterbiCUDA();
	ViterbiCUDA(size_t inputNum);
	~ViterbiCUDA();
	
	void run(encPack_t* input_h, decPack_t* output_h, size_t messageLen, float* kernelTime = nullptr);

	size_t getInputSize(size_t inputNum);
	size_t getMessageLen(size_t inputNum);
	size_t getOutputSize(size_t inputNum);

private:
	decPack_t* pathPrev_d;
	decPack_t* dec_d;
	encPack_t* enc_d;
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