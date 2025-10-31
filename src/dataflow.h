#pragma once

#include <variant>
#include <vector>
#include <random>
#include <optional>
#include <iostream>
#include <memory>
#include <algorithm>
#include "viterbi.h"

// Data type used between elements
enum class Bit : uint8_t { OFF=0, ON=1 };
using Bits = std::vector<Bit>;
using Soft = std::vector<uint8_t>;
using Reals = std::vector<float>;
template<Metric metricType> using BitsPack = std::vector<pack_t<metricType>>; // packed bits (8 bits per byte)
using Data = std::variant<Bits, Soft, Reals, BitsPack<Metric::B16>, BitsPack<Metric::B32>>;
using OptData = std::optional<Data>;

// Base class
struct ComputeElement {
    bool is_probed = false;
    virtual ~ComputeElement() = default;
    // If in == std::nullopt, element should generate its own data (e.g. RandBitGen)
    virtual Data process(const OptData& in) = 0;

    // Method to enable probing on this element
    ComputeElement& probe() {
        is_probed = true;
        return *this;
    }
};

// The result of a pipeline run, containing the final output and any probed intermediate values.
struct PipelineResult {
    Data final_output;
    std::vector<Data> probed_outputs;
};

// Pipeline container
struct Pipeline {
    std::vector<ComputeElement*> elems;
    Pipeline() = default;
    Pipeline& add(ComputeElement& e){ elems.push_back(&e); return *this; }

    PipelineResult run() {
        OptData cur = std::nullopt;
        std::vector<Data> probes;
        for(auto e : elems){
            cur = e->process(cur);
            if(e->is_probed) {
                probes.push_back(*cur);
            }
        }
        if(!cur.has_value()) throw std::runtime_error("Pipeline produced no output");
        return {*cur, probes};
    }
};

// chaining operators
inline Pipeline operator|(ComputeElement& a, ComputeElement& b){
    Pipeline p; p.add(a).add(b); return p;
}

// Takes the pipeline by value to allow chaining with temporary objects.
// The 'p' parameter will be move-constructed, which is very efficient.
inline Pipeline operator|(Pipeline p, ComputeElement& b){
    p.add(b); return p;
}

// ----- Example building blocks -----

// 1) Random bit generator
struct RandBitGen : ComputeElement {
    size_t n;
    std::mt19937 rng;
    RandBitGen(size_t n_, unsigned seed=0) : n(n_), rng(seed) {}
    Data process(const OptData& in) override {
        Bits out; out.reserve(n);
        std::uniform_int_distribution<int> d(0,1);
        for(size_t i=0;i<n;++i) out.push_back(d(rng) ? Bit::ON : Bit::OFF);
        return out;
    }
};

// 2) Convolutional encoder (simple version matching convEnc in vit_main.cu)
struct ConvolutionalEncoder : ComputeElement {
    int CLength;
    uint32_t polyn0, polyn1;
    ConvolutionalEncoder(int CLength_, uint32_t p0, uint32_t p1) : CLength(CLength_), polyn0(p0), polyn1(p1) {}
    Data process(const OptData& in) override {
        if(!in) throw std::runtime_error("ConvolutionalEncoder expects input bits");
        Bits bits = std::get<Bits>(*in);
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
struct AddNoise : ComputeElement {
    float stddev;
    unsigned seed;
    AddNoise(float stddev_, unsigned seed_=0) : stddev(stddev_), seed(seed_) {}
    Data process(const OptData& in) override {
        if(!in) throw std::runtime_error("AddNoise expects input bits");
        Bits bits = std::get<Bits>(*in);
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
struct SoftDecisionPacker : ComputeElement {
    Input cfg;
    float scale; // optional scaling applied before quantization

    SoftDecisionPacker(Input cfg_, float scale_ = 1.0f) : cfg(cfg_), scale(scale_) {}

private:
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
    Data process(const OptData& in) override {
        if(!in) throw std::runtime_error("SoftDecisionPacker expects input reals");
        const Reals& src = std::get<Reals>(*in);
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
