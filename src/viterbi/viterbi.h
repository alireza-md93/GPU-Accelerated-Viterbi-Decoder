#pragma once

#include <cstdint>
#include <type_traits>
#include <cuda_fp16.h>

constexpr int CHANNEL_SHIFT = 0;
constexpr int METRIC_SHIFT = 4;
constexpr int DECODE_SHIFT = 8;
constexpr int COMP_SHIFT = 12;

constexpr int CHANNEL_MASK = (0xf << 0);
constexpr int METRIC_MASK = (0xf << 4);
constexpr int DECODE_MASK = (0xf << 8);
constexpr int COMP_MASK =  (0xf <<12);

enum ChannelIn {HARD=(0x0<<CHANNEL_SHIFT), SOFT4=(0x1<<CHANNEL_SHIFT), SOFT8=(0x2<<CHANNEL_SHIFT), SOFT16=(0x3<<CHANNEL_SHIFT), FP32=(0x4<<CHANNEL_SHIFT)};
enum Metric {M_B32=(0x0<<METRIC_SHIFT), M_B16=(0x1<<METRIC_SHIFT), M_FP16=(0x2<<METRIC_SHIFT)};
enum DecodeOut {O_B32=(0x0<<DECODE_SHIFT), O_B16=(0x1<<DECODE_SHIFT)};
enum CompMode{REG=(0x0<<COMP_SHIFT), DPX=(0x1<<COMP_SHIFT)};

template<int options>
struct OptionsValid{
	static constexpr bool value = 
		((options & CHANNEL_MASK) == ChannelIn::SOFT8 &&
		 (options & METRIC_MASK) == Metric::M_FP16) ||

		((options & CHANNEL_MASK) == ChannelIn::SOFT16 &&
		 (options & METRIC_MASK) == Metric::M_FP16) ||

		((options & CHANNEL_MASK) == ChannelIn::SOFT16 &&
		 (options & METRIC_MASK) == Metric::M_B16) ||

		((options & METRIC_MASK) == Metric::M_FP16 &&
		 (options & COMP_MASK) == CompMode::DPX) ? false : true;

	// static constexpr bool value = 
	// 	((options & CHANNEL_MASK) == ChannelIn::HARD && 
	// 	 (options & METRIC_MASK) == Metric::M_B32 &&
	// 	 (options & DECODE_MASK) == DecodeOut::O_B32) ? true : false;
};

template<int options = 0, bool enable = OptionsValid<options>::value>
class ViterbiCUDA;

template<int options>
struct ViterbiCUDA<options, false>{
	public:
	
	static constexpr ChannelIn inputType = static_cast<ChannelIn>(options & CHANNEL_MASK);
	static constexpr Metric metricType = static_cast<Metric>(options & METRIC_MASK);
	static constexpr DecodeOut outputType = static_cast<DecodeOut>(options & DECODE_MASK);
	static constexpr CompMode compMode = static_cast<CompMode>(options & COMP_MASK);

	using metric_t = 	std::conditional_t<metricType == Metric::M_B16, int16_t, 
						std::conditional_t<metricType == Metric::M_B32, int32_t,
						std::conditional_t<metricType == Metric::M_FP16, __half, float>>>;
	using decPack_t = 	std::conditional_t<outputType == DecodeOut::O_B16, uint16_t, uint32_t>;
	using encPack_t = std::conditional_t<inputType == ChannelIn::FP32, float, int32_t>; // packed encoded input
	
	static constexpr int constLen = 7; //constraint length
	static constexpr int polyn1 = 0171; //polynomial 1
	static constexpr int polyn2 = 0133; //polynomial 2

	static constexpr int roundup(int a, int b) { if(a <= 0) return 0; else return ((a + b - 1) / b * b); };
	static constexpr size_t roundup(size_t a, size_t b) { if(a <= 0) return 0; else return ((a + b - 1) / b * b); };
	static constexpr int bitsPerMetric = 	(metricType == Metric::M_B16) ? 16 : 
											(metricType == Metric::M_B32) ? 32 : 11;
	static constexpr int bitsPerPack = 		(outputType == DecodeOut::O_B16) ? 16 : 32;
	static constexpr int extraL_raw = 32;
	static constexpr int extraR_raw = 32;
	static constexpr int slideSize_raw = 32;
	static constexpr int extraL = roundup(extraL_raw, bitsPerPack) - (constLen - 1);
	static constexpr int extraR = roundup(extraR_raw, bitsPerPack) + (constLen - 1);
	static constexpr int slideSize = roundup(slideSize_raw, bitsPerPack);
	static constexpr int forwardLen = extraL + slideSize + extraR;
	static constexpr int bmMemWidth = 32;//32/4;
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
	
};

template<int options>
class ViterbiCUDA<options, true> : public ViterbiCUDA<options, false>{
	public:

	using ViterbiCUDA<options, false>::inputType;
	using ViterbiCUDA<options, false>::metricType;
	using ViterbiCUDA<options, false>::outputType;
	using ViterbiCUDA<options, false>::compMode;

	using typename ViterbiCUDA<options, false>::metric_t;
	using typename ViterbiCUDA<options, false>::decPack_t;
	using typename ViterbiCUDA<options, false>::encPack_t;

	using ViterbiCUDA<options, false>::roundup;

	using ViterbiCUDA<options, false>::constLen;
	using ViterbiCUDA<options, false>::polyn1;
	using ViterbiCUDA<options, false>::polyn2;

	using ViterbiCUDA<options, false>::bitsPerMetric;
	using ViterbiCUDA<options, false>::bitsPerPack;
	using ViterbiCUDA<options, false>::extraL_raw;
	using ViterbiCUDA<options, false>::extraR_raw;
	using ViterbiCUDA<options, false>::slideSize_raw;
	using ViterbiCUDA<options, false>::extraL;
	using ViterbiCUDA<options, false>::extraR;
	using ViterbiCUDA<options, false>::slideSize;
	using ViterbiCUDA<options, false>::forwardLen;
	using ViterbiCUDA<options, false>::bmMemWidth;
    using ViterbiCUDA<options, false>::blockDimY;
    using ViterbiCUDA<options, false>::FPprecision;
	using ViterbiCUDA<options, false>::encDataPerPack;
	using ViterbiCUDA<options, false>::encDataWidth;
	

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