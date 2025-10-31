
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <memory>

#include "viterbi.h"
#include "dataflow.h"


void parseArg(int argc, char *argv[], int& messageLen, float& snr);

template<Metric metricType>
struct ViterbiDecoder : ComputeElement {
private:
	std::unique_ptr<ViterbiCUDA<metricType>> viterbi;

public:
    // In real project this would call your GPU viterbi_run
    ViterbiDecoder(): viterbi(new ViterbiCUDA<metricType>()) {}
	ViterbiDecoder(int messageLen): viterbi(new ViterbiCUDA<metricType>(messageLen)) {}
	~ViterbiDecoder() = default;
    Data process(const OptData& in) override {
        if(!in) throw std::runtime_error("ViterbiDecoder expects input reals");
        Reals soft = std::get<Reals>(*in);
        BitsPack<metricType> out; out.resize(viterbi->getOutputSize(soft.size())/sizeof(pack_t<metricType>)); // assume soft contains coded pairs -> decode to bits (simple approach)
		float gpuKernelTime;
		viterbi->run(soft.data(), out.data(), soft.size(), &gpuKernelTime);
		printf("Viterbi GPU kernel time: %f ms\n", gpuKernelTime);
        return out;
    }
};
template struct ViterbiDecoder<Metric::B16>;
template struct ViterbiDecoder<Metric::B32>;

int main(int argc, char *argv[]) {

    // get parameters from command line
    int messageLen;
	float snr;
    parseArg(argc, argv, messageLen, snr);
	constexpr Metric metricType = Metric::B32;
	constexpr Input inputType = Input::SOFT8;
	int messageLen_ext = messageLen + ViterbiCUDA<metricType>::extraL + ViterbiCUDA<metricType>::extraR;
	int packingWidth = metricType;
	
	
	// --- Dataflow Pipeline Setup ---
	std::random_device rd;
	RandBitGen randGen(messageLen_ext, rd());
    ConvolutionalEncoder convEnc(ViterbiCUDA<metricType>::constLen, ViterbiCUDA<metricType>::polyn1, ViterbiCUDA<metricType>::polyn2);
    AddNoise noise(pow(10, -snr/20.0));
    SoftDecisionPacker packer(inputType);
    ViterbiDecoder<metricType> viterbi;

    // Build the pipeline, probing the output of the noise adder.
    Pipeline pipe = randGen.probe() | convEnc | noise | viterbi;
    PipelineResult result = pipe.run();
	int BENs;//bit error number; 
	double BERs;//bit error rate
	int minInd, maxInd;
	BENs = 0;
	minInd=-1; //minimum index of errors
	maxInd = 0; //maximum index of errors
    size_t decodedBitsLen = std::get<BitsPack<metricType>>(result.final_output).size()*packingWidth;
	for(int i=0; i<decodedBitsLen; i++){
		bool decodedBit = (std::get<BitsPack<metricType>>(result.final_output)[i/packingWidth] & (1U<<((packingWidth-1)-(i%packingWidth)))) == 0 ? false : true;
		bool genBit = (std::get<Bits>(result.probed_outputs[0])[i+ViterbiCUDA<metricType>::extraL] == Bit::ON) ? true : false;
		if(decodedBit != genBit){
			// printf(":%d\n", i);
			BENs++;
			maxInd = i;
			if(minInd == -1)
				minInd = i;
		}
	}
	BERs = (double)BENs / messageLen;

    std::cout << "Pipeline executed." << std::endl;
    std::cout << "Final BEN: " << BENs << std::endl;
	printf("min:%d\tmax:%d\n", minInd, maxInd);
	printf("BEN:%d   \tBER:%f\n", BENs, BERs);
    std::cout << "Number of probed outputs: " << result.probed_outputs.size() << std::endl;
    if (!result.probed_outputs.empty()) {
        std::cout << "Probed output type index: " << result.probed_outputs[0].index() << std::endl;
    }
    // --- End of Pipeline Section ---
	
	
    return 0;
}
//-----------------------------------------------------------------------------
void parseArg(int argc, char *argv[], int& messageLen, float& snr) {
    // Set default values
    messageLen = 32000000;
    snr = 15.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -n, --num <integer>      Set the message length.\n"
                      << "  -s, --snr <float>        Set the Signal-to-Noise Ratio (SNR).\n"
                      << "  -h, --help               Display this help message.\n";
            exit(0);
        } else if ((arg == "-n" || arg == "--num") && i + 1 < argc) {
            try {
                messageLen = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid argument for " << arg << ". Please provide an integer." << std::endl;
                exit(1);
            }
        } else if ((arg == "-s" || arg == "--snr") && i + 1 < argc) {
            try {
                snr = std::stof(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid argument for " << arg << ". Please provide a float." << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Error: Unknown or incomplete argument: " << arg << std::endl;
            exit(1);
        }
    }
}
