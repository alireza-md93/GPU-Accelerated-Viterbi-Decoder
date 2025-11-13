#pragma once

#include "dataflow.h"
#include "viterbi.h"

#include <vector>
#include <random>
#include <memory>

// Data type used between elements
enum class Bit : uint8_t { OFF=0, ON=1 };
using Bits = std::vector<Bit>;
using Soft = std::vector<uint8_t>;
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
    AddNoise(float stddev_, unsigned seed_=0) : stddev(stddev_), seed(seed_) {}
    std::any process(const OptData& in) override {
        if(!in) throw std::runtime_error("AddNoise expects input bits");
        Bits bits = std::any_cast<Bits>(*in);
        Reals out; out.reserve(bits.size());
        std::mt19937 rng(seed);
        std::normal_distribution<float> d(0.0f, stddev);
        for(Bit b : bits){
            float base = (b == Bit::ON) ? 1.0f : -1.0f;
            out.push_back(base + d(rng));
        }
        return out;
    }
};

// 4) SoftDecisionPacker (example: pack floats into same vector or perform quantization)
class SoftDecisionPacker : public ComputeElement {
    private:
    Input cfg;
    float scale; // optional scaling applied before quantization

    static inline uint8_t pack_hard_byte(const float *vals){
        uint8_t b = 0;
        for(size_t k=0; k<8; ++k){
            b <<= 1;
            b |= (vals[k] > 0.0f) ? 1u : 0u;
        }
        return b;
    }

    static inline uint8_t quant4(float v){
        // convert to signed int4 in range [-8,7], store as two's complement nibble
        int q = (int)std::lrintf(v); // round to nearest int
        if(q < -8) q = -8;
        if(q > 7) q = 7;
        return static_cast<uint8_t>(q & 0xF);
    }

    static inline uint8_t quant8(float v){
        // convert to signed int8 range [-128,127], store as uint8 (two's complement)
        int q = (int)std::lrintf(v);
        if(q < -128) q = -128;
        if(q > 127) q = 127;
        return static_cast<uint8_t>(q & 0xFF);
    }

    public:
    SoftDecisionPacker(Input cfg_, float scale_ = 1.0f) : cfg(cfg_), scale(scale_) {}
    std::any process(const OptData& in) override {
        if(!in) throw std::runtime_error("SoftDecisionPacker expects input reals");
        const Reals& src = std::any_cast<Reals>(*in);
        size_t n = src.size();
        Soft out;

        if(cfg == Input::HARD){
            // pack 8 floats -> 1 byte
            out.reserve((n + 7) / 8);
            for(size_t i=0; i<n; i += 8){
                size_t available = std::min<size_t>(8, n - i);
                float tmp[8];
                for(size_t k=0;k<available;++k) tmp[k] = src[i+k] * scale;
                for(size_t k=available;k<8;++k) tmp[k] = 0.0f;
                // remaining tmp entries ignored (treated as <=0)
                out.push_back(pack_hard_byte(tmp));
            }
            return out;
        }
        else if(cfg == Input::SOFT4){
            // pack every 2 floats -> one uint8 (high nibble = first float, low nibble = second)
            out.reserve((n + 1) / 2);
            for(size_t i=0; i<n; i += 2){
                float v0 = (i < n)     ? src[i]   * scale : 0.0f;
                float v1 = (i+1 < n)   ? src[i+1] * scale : 0.0f;
                uint8_t hi = quant4(v0);
                uint8_t lo = quant4(v1);
                out.push_back(static_cast<uint8_t>((hi << 4) | (lo & 0x0F)));
            }
            return out;
        }
        else if(cfg == Input::SOFT8){
            // one float -> one uint8 (signed int8 stored as uint8)
            out.reserve(n);
            for(size_t i=0;i<n;++i){
                out.push_back(quant8(src[i] * scale));
            }
            return out;
        }

        throw std::runtime_error("Unsupported SoftDecisionPacker configuration");
    }
};

// 5) Viterbi Decoder wrapper
template<Metric metricType>
struct ViterbiDecoder : ComputeElement {
private:
	std::unique_ptr<ViterbiCUDA<metricType>> viterbi;

public:
    using decPack_t = typename ViterbiCUDA<metricType>::decPack_t;
    using decVec_t = std::vector<typename ViterbiCUDA<metricType>::decPack_t>; // packed bits (8 bits per byte)
    static constexpr int bitsPerPack = ViterbiCUDA<metricType>::bitsPerPack;
    // In real project this would call your GPU viterbi_run
    ViterbiDecoder(): viterbi(new ViterbiCUDA<metricType>()) {}
	ViterbiDecoder(int messageLen): viterbi(new ViterbiCUDA<metricType>(messageLen)) {}
	~ViterbiDecoder() = default;
    std::any process(const OptData& in) override {
        if(!in) throw std::runtime_error("ViterbiDecoder expects input reals");
        Reals soft = std::any_cast<Reals>(*in);
        decVec_t out; 
        out.resize(viterbi->getOutputSize(soft.size())/sizeof(decPack_t));
		float gpuKernelTime;
		viterbi->run(soft.data(), out.data(), soft.size(), &gpuKernelTime);
		printf("Viterbi GPU kernel time: %f ms\n", gpuKernelTime);
        return out;
    }
};
template struct ViterbiDecoder<Metric::B16>;
template struct ViterbiDecoder<Metric::B32>;
