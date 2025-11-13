#pragma once

#include <any>
#include <vector>
#include <optional>
// #include <iostream>

using OptData = std::optional<std::any>;

// Base class
class ComputeElement {
    private:
    bool is_probed;

    public:
    ComputeElement() : is_probed(false) {}
    virtual ~ComputeElement() = default;
    // If in == std::nullopt, element should generate its own data (e.g. RandBitGen)
    virtual std::any process(const OptData& in) = 0;

    // Method to enable probing on this element
    ComputeElement& probe() {
        is_probed = true;
        return *this;
    }

    bool isProbed() const { return is_probed; }
};

// The result of a pipeline run, containing the final output and any probed intermediate values.
struct PipelineResult {
    std::any final_output;
    std::vector<std::any> probed_outputs;
};

// Pipeline container
class Pipeline {
    private:
    std::vector<ComputeElement*> elems;

    public:
    Pipeline() = default;
    Pipeline& add(ComputeElement& e){ elems.push_back(&e); return *this; }

    PipelineResult run() {
        OptData cur = std::nullopt;
        std::vector<std::any> probes;
        for(auto e : elems){
            // std::cout << "Processing element: " << typeid(*e).name() << std::endl;
            cur = e->process(cur);
            if(e->isProbed()) {
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