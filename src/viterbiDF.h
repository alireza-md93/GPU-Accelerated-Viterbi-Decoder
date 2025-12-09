#pragma once

#include "dataflow.h"
#include "viterbi.h"

#include <vector>
#include <random>
#include <memory>
#include <map>
#include <functional>

// Data type used between elements
enum class Bit : uint8_t { OFF=0, ON=1 };
using Bits = std::vector<Bit>;
using soft_t = int32_t;
using Soft = std::vector<soft_t>;
using Reals = std::vector<float>;

// 1) Random bit generator
class RandBitGen : public ComputeElement {
    private:
    size_t n;
    std::mt19937 rng;

    public:
    RandBitGen(size_t n_, unsigned seed=0) : n(n_), rng(seed) {}
    std::any process(const OptData& in) override {
        Bits out; out.reserve(n);
        std::uniform_int_distribution<int> d(0,1);
        for(size_t i=0;i<n;++i) out.push_back(d(rng) ? Bit::ON : Bit::OFF);
        return out;
    }
};

// 2) Convolutional encoder (simple version matching convEnc in vit_main.cu)
class ConvolutionalEncoder : public ComputeElement {
    private:
    int CLength;
    uint32_t polyn0, polyn1;

    public:
    ConvolutionalEncoder(int CLength_, uint32_t p0, uint32_t p1) : CLength(CLength_), polyn0(p0), polyn1(p1) {}
    std::any process(const OptData& in) override {
        if(!in) throw std::runtime_error("ConvolutionalEncoder expects input bits");
        Bits bits = std::any_cast<Bits>(*in);
        size_t n = bits.size();
        Bits coded; coded.resize(2*n);
        uint32_t buffer = 0;
        for(size_t i=0;i<n;i++){
            buffer >>= 1;
            buffer |= (static_cast<uint32_t>(bits[i] == Bit::ON) << (CLength-1));
            uint32_t temp = buffer & polyn0;
            uint8_t out0 = 0;
            for(int cnt=0; cnt<CLength; ++cnt){ out0 ^= (temp & 1); temp >>= 1; }
            temp = buffer & polyn1;
            uint8_t out1 = 0;
            for(int cnt=0; cnt<CLength; ++cnt){ out1 ^= (temp & 1); temp >>= 1; }
            coded[2*i+0] = out0 ? Bit::ON : Bit::OFF;
            coded[2*i+1] = out1 ? Bit::ON : Bit::OFF;
        }
        return coded;
    }
};

// 3) Add Gaussian noise and produce soft floats
class AddNoise : public ComputeElement {
    private:
    float stddev;
    unsigned seed;

    public:
    AddNoise(float stddev_=std::numeric_limits<float>::infinity(), unsigned seed_=0) : stddev(stddev_), seed(seed_) {}
    std::any process(const OptData& in) override {
        if(!in) throw std::runtime_error("AddNoise expects input bits");
        Bits bits = std::any_cast<Bits>(*in);
        Reals out; out.reserve(bits.size());
        std::mt19937 rng(seed);
        std::normal_distribution<float> d(0.0f, stddev);
        if(stddev == std::numeric_limits<float>::infinity()){
            // no noise case
            for(Bit b : bits){
                float val = (b == Bit::ON) ? 1.0f : -1.0f;
                out.push_back(val);
            }
        }
        else{
            // noise case
            for(Bit b : bits){
                float base = (b == Bit::ON) ? 1.0f : -1.0f;
                out.push_back(base + d(rng));
            }
        }
        return out;
    }
};

// 4) SoftDecisionPacker (example: pack floats into same vector or perform quantization)
class SoftDecisionPacker : public ComputeElement {
    private:
    ChannelIn cfg;
    float scale;
    int dataPerPack;
    int packLength;

    inline static std::map<ChannelIn, std::function<soft_t(float)>> quantFuncs= {
        {ChannelIn::HARD,   [](float v) -> soft_t { return (v > 0.0f) ? 1 : 0; }},
        {ChannelIn::SOFT4,  [](float v) -> soft_t {
            int q = static_cast<int>(std::lrintf(v));
            if(q < -8) q = -8;
            if(q > 7) q = 7;
            return static_cast<soft_t>(q & 0xF);
        }},
        {ChannelIn::SOFT8,  [](float v) -> soft_t {
            int q = static_cast<int>(std::lrintf(v));
            if(q < -128) q = -128;
            if(q > 127) q = 127;
            return static_cast<soft_t>(q & 0xFF);
        }},
        {ChannelIn::SOFT16, [](float v) -> soft_t {
            long q = static_cast<long>(std::lrintf(v));
            if(q < -32768) q = -32768;
            if(q > 32767) q = 32767;
            return static_cast<soft_t>(q&0xFFFF);
        }}
    };

    public:
    SoftDecisionPacker(ChannelIn cfg_, float scale_=1.0f) : cfg(cfg_), scale(scale_) {
        size_t elementSize = sizeof(typename Soft::value_type);
        switch(cfg){
            case ChannelIn::HARD:    packLength = 1;  dataPerPack = elementSize * 8; break;
            case ChannelIn::SOFT4:   packLength = 4;  dataPerPack = elementSize * 2; break;
            case ChannelIn::SOFT8:   packLength = 8;  dataPerPack = elementSize;     break;
            case ChannelIn::SOFT16:  packLength = 16; dataPerPack = elementSize / 2; break;
        }
    }
    std::any process(const OptData& in) override {
        if(!in) throw std::runtime_error("SoftDecisionPacker expects input reals");
        const Reals& src = std::any_cast<Reals>(*in);
        if(cfg == ChannelIn::FP32){
            if(scale == 1.0){
                return src;
            } else {
                Reals out;
                out.reserve(src.size());
                for(float v : src){
                    out.push_back(v * scale);
                }
                return out;
            }
        }
        std::function<soft_t(float)> quant = quantFuncs.at(cfg);
        size_t n = src.size();

        Soft out;
        out.reserve(n / dataPerPack);
        for(size_t i=0; i<n; i += dataPerPack){
            uint32_t b = 0;
            for(size_t j=i; j<i+dataPerPack; ++j){
                b <<= packLength;
                b |= quant(src[j]*scale);
            }
            out.push_back(static_cast<soft_t>(b));
        }
        return out;
    }
};

// 5) Viterbi Decoder wrapper
template<Metric metricType, ChannelIn inputType>
struct ViterbiDecoder : ComputeElement {
private:
	std::unique_ptr<ViterbiCUDA<metricType, inputType>> viterbi;

public:
    using decPack_t = typename ViterbiCUDA<metricType, inputType>::decPack_t;
    using decVec_t = std::vector<typename ViterbiCUDA<metricType, inputType>::decPack_t>; // packed bits (8 bits per byte)
    using encPack_t = typename ViterbiCUDA<metricType, inputType>::encPack_t;
    static constexpr int bitsPerPack = ViterbiCUDA<metricType, inputType>::bitsPerPack;
    static constexpr int encDataPerPack = ViterbiCUDA<metricType, inputType>::encDataPerPack;

    // In real project this would call your GPU viterbi_run
    ViterbiDecoder(): viterbi(new ViterbiCUDA<metricType, inputType>()) {}
	ViterbiDecoder(int messageLen): viterbi(new ViterbiCUDA<metricType, inputType>(messageLen)) {}
	~ViterbiDecoder() = default;
    std::any process(const OptData& in) override {
        if(!in) throw std::runtime_error("ViterbiDecoder expects input reals");
        std::vector<encPack_t> soft = std::any_cast<std::vector<encPack_t>>(*in); 
        decVec_t out; 
        size_t decInputNum = soft.size() * encDataPerPack;
        out.resize(viterbi->getOutputSize(decInputNum)/sizeof(decPack_t));
		float gpuKernelTime;
		viterbi->run(soft.data(), out.data(), decInputNum, &gpuKernelTime);
        setStatus("GPU kernel time", gpuKernelTime);
        return out;
    }
    std::string getStatusString(const std::string& key) const override {
        if(key == "GPU kernel time") {
            const auto valueAny = getStatus(key);
            float value = std::any_cast<float>(valueAny);
            std::stringstream ss;
            if(value < 1.0f)            {ss << std::fixed << std::setprecision(3) << value*1000.0f << " us";}
            else if(value < 1000.0f)    {ss << std::fixed << std::setprecision(3) << value << " ms";}
            else                        {ss << std::fixed << std::setprecision(3) << value/1000.0f << " s";}
            return ss.str();
        }
        return ComputeElement::getStatusString(key);
    }
};
template struct ViterbiDecoder<Metric::B16, ChannelIn::HARD>;
template struct ViterbiDecoder<Metric::B16, ChannelIn::SOFT4>;
template struct ViterbiDecoder<Metric::B16, ChannelIn::SOFT8>;
//--- never to be enabled ---// template struct ViterbiDecoder<Metric::B16, ChannelIn::SOFT16>;
template struct ViterbiDecoder<Metric::B16, ChannelIn::FP32>;

template struct ViterbiDecoder<Metric::B32, ChannelIn::HARD>;
template struct ViterbiDecoder<Metric::B32, ChannelIn::SOFT4>;
template struct ViterbiDecoder<Metric::B32, ChannelIn::SOFT8>;
template struct ViterbiDecoder<Metric::B32, ChannelIn::SOFT16>;
template struct ViterbiDecoder<Metric::B32, ChannelIn::FP32>;

template struct ViterbiDecoder<Metric::FP16, ChannelIn::HARD>;
template struct ViterbiDecoder<Metric::FP16, ChannelIn::SOFT4>;
//--- never to be enabled ---// template struct ViterbiDecoder<Metric::FP16, ChannelIn::SOFT8>;
//--- never to be enabled ---// template struct ViterbiDecoder<Metric::FP16, ChannelIn::SOFT16>;
template struct ViterbiDecoder<Metric::FP16, ChannelIn::FP32>;