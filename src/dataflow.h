#pragma once

#include <any>
#include <vector>
#include <map>
#include <string>
#include <optional>
#include <chrono>
#include <iostream>
#include <typeinfo>
#include <iomanip>

using OptData = std::optional<std::any>;

// Base class
class ComputeElement {
private:
    bool is_probed;

protected:
    std::map<std::string, std::any> status;

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

    void setStatus(const std::string& key, std::any value) {
        status[key] = std::move(value);
    }

    std::any getStatus(const std::string& key) const {
        return status.at(key);
    }

    const std::map<std::string, std::any>& getStatusMap() const {
        return status;
    }

    virtual std::string getStatusString(const std::string& key) const {
        return "(Not printable)";
    }


    std::string getStatusStringAll(const std::string& key) const {
        if(key == "Elapsed run time") {
            const auto& value = getStatus(key);
            auto us = std::any_cast<std::chrono::microseconds>(value).count();
            std::stringstream ss;
            if (us > 1000000) {
                ss << std::fixed << std::setprecision(2) << us / 1000000.0 << " s";
            } else if (us > 1000) {
                ss << std::fixed << std::setprecision(2) << us / 1000.0 << " ms";
            } else {
                ss << us << " us";
            }
            return ss.str();
        }

        return getStatusString(key);
    }
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
        for(auto e : elems) {
            auto start = std::chrono::high_resolution_clock::now();
            // std::cout << "Processing element: " << typeid(*e).name() << std::endl;
            cur = e->process(std::move(cur));
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            e->setStatus("Elapsed run time", duration);

            if(e->isProbed()) {
                probes.push_back(*cur);
            }
        }
        if(!cur.has_value()) throw std::runtime_error("Pipeline produced no output");
        return {*cur, probes};
    }

    void printStatus() const {
        std::cout << "--- Pipeline Status ---\n";
        int i = 0;
        for (const auto e : elems) {
            std::cout << "Element " << i++ << " (type: " << typeid(*e).name() << "):\n";
            const auto& status_map = e->getStatusMap();
            if (status_map.empty()) {
                std::cout << "  - No status information.\n";
            } else {
                for (const auto& [key, value] : status_map) {
                    std::cout << "  - " << key << ": " << e->getStatusStringAll(key) << "\n";
                }
            }
        }
        std::cout << "--- End of Status ---\n";
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