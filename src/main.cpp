
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include "viterbiDF.h"

void parseArg(int argc, char *argv[], int& messageLen, float& snr, Metric& metricType, ChannelIn& inputType);

template<Metric metricType, ChannelIn inputType>
void runPipeline(int messageLen, float snr, int& BENs, bool showStatus=false);
void runPipelineWrapper(Metric metricType, ChannelIn inputType, int messageLen, float snr, int& BENs, bool showStatus=false);

//-----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    // get parameters from command line
    Metric metricType;
	ChannelIn inputType;
    int messageLen;
	float snr;
    parseArg(argc, argv, messageLen, snr, metricType, inputType);

    std::cout << "Message Length: " << messageLen << std::endl;
    std::cout << "SNR: " << snr << " dB" <<  std::endl;
    std::cout << "Metric Type: " << ((metricType == Metric::B16) ? "16-bit" : "32-bit") << std::endl;
    std::cout << "Input Channel Type: ";
    switch(inputType){
        case ChannelIn::HARD:
            std::cout << "Hard Decision" << std::endl;
            break;
        case ChannelIn::SOFT4:
            std::cout << "4-bit Soft Decision" << std::endl;
            break;
        case ChannelIn::SOFT8:
            std::cout << "8-bit Soft Decision" << std::endl;
            break;
        case ChannelIn::SOFT16:
            std::cout << "16-bit Soft Decision" << std::endl;
            break;
        case ChannelIn::FP32:
            std::cout << "32-bit Floating Point" << std::endl;
            break;
        default:
            std::cout << "Unknown Type" << std::endl;
            break;
    }
    std::cout << std::endl;

    // --- Dataflow Pipeline Execution Section ---
    int BENs; //bit error number
    double BERs; //bit error rate
    runPipelineWrapper(metricType, inputType, messageLen, snr, BENs, true);
	
	BERs = (double)BENs / messageLen;

    std::cout << "Pipeline executed." << std::endl;
    std::cout << "Final results -> BEN: " << BENs << "   BER: " << BERs << std::endl;
    // --- End of Pipeline Section ---
	
	
    return 0;
}

//-----------------------------------------------------------------------------

void runPipelineWrapper(Metric metricType, ChannelIn inputType, int messageLen, float snr, int& BENs, bool showStatus){
    if(metricType == Metric::B16){
        if(inputType == ChannelIn::HARD)           runPipeline<Metric::B16, ChannelIn::HARD>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT4)     runPipeline<Metric::B16, ChannelIn::SOFT4>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT8)     runPipeline<Metric::B16, ChannelIn::SOFT8>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT16)    runPipeline<Metric::B16, ChannelIn::SOFT16>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::FP32)      runPipeline<Metric::B16, ChannelIn::FP32>(messageLen, snr, BENs, showStatus);
    }
    else if(metricType == Metric::B32){
        if(inputType == ChannelIn::HARD)           runPipeline<Metric::B32, ChannelIn::HARD>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT4)     runPipeline<Metric::B32, ChannelIn::SOFT4>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT8)     runPipeline<Metric::B32, ChannelIn::SOFT8>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT16)    runPipeline<Metric::B32, ChannelIn::SOFT16>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::FP32)      runPipeline<Metric::B32, ChannelIn::FP32>(messageLen, snr, BENs, showStatus);
    }
    else{
        std::cerr << "Unsupported metric type." << std::endl;
    }
}

template<Metric metricType, ChannelIn inputType>
void runPipeline(int messageLen, float snr, int& BENs, bool showStatus){
    constexpr int CL = ViterbiCUDA<metricType, inputType>::constLen;
    constexpr int polyn1 = ViterbiCUDA<metricType, inputType>::polyn1;
    constexpr int polyn2 = ViterbiCUDA<metricType, inputType>::polyn2;
    constexpr int extraL = ViterbiCUDA<metricType, inputType>::extraL;
    constexpr int extraR = ViterbiCUDA<metricType, inputType>::extraR;
    using decVec_t = typename ViterbiDecoder<metricType, inputType>::decVec_t;

	int bitsPerPack = ViterbiDecoder<metricType, inputType>::bitsPerPack;
	
	// --- Dataflow Pipeline Setup ---
	std::random_device rd;
	// RandBitGen randGen(messageLen, 0);
	RandBitGen randGen (messageLen, rd());
    ConvolutionalEncoder convEnc(CL, polyn1, polyn2);
    AddNoise noise(pow(10, -snr/5.0), rd());
    // AddNoise noise(std::numeric_limits<float>::infinity());
    SoftDecisionPacker packer(inputType);
    ViterbiDecoder<metricType, inputType> viterbi;

    // Build the pipeline, probing the output of the noise adder.
    Pipeline pipe = randGen.probe() | convEnc | noise | packer | viterbi;
    PipelineResult result = pipe.run();
    // --- End of Pipeline Setup ---

    if(showStatus){
        std::cout << std::endl;
        pipe.printStatus();
        std::cout << std::endl;
    }
    
    // --- BER Calculation Section ---
	// int minInd, maxInd;
	BENs = 0;
	// minInd = -1; //minimum index of errors
	// maxInd = 0; //maximum index of errors
    size_t decodedBitsLen = std::any_cast<decVec_t>(result.final_output).size()*bitsPerPack;
    decVec_t decodedBitsVector = std::any_cast<decVec_t>(result.final_output);
	Bits genBitsVector = std::any_cast<Bits>(result.probed_outputs[0]);
    for(int i=0; i<decodedBitsLen; i++){
		bool decodedBit = (decodedBitsVector[i/bitsPerPack] & (1U<<((bitsPerPack-1)-(i%bitsPerPack)))) == 0 ? false : true;
		bool genBit = (genBitsVector[i+extraL] == Bit::ON) ? true : false;
        if(decodedBit != genBit){
			BENs++;
			// maxInd = i;
			// if(minInd == -1)
			// 	minInd = i;
		}
	}
}

void parseArg(int argc, char *argv[], int& messageLen, float& snr, Metric& metricType, ChannelIn& inputType) {
    // Set default values
    messageLen = 32000000;
    snr = 15.0;
    metricType = Metric::B32;
    inputType = ChannelIn::HARD;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -n, --num <integer>      Set the message length.\n"
                      << "  -s, --snr <float>        Set the Signal-to-Noise Ratio (SNR).\n"
                      << "  -m, --metric <type>      Set the metric type (16bit|16 or 32bit|32).\n"
                      << "  -i, --input <type>       Set the input channel type (HARD|h, SOFT4|s4, SOFT8|s8, SOFT16|s16, FP32|f).\n"
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
        } else if ((arg == "-m" || arg == "--metric") && i + 1 < argc) {
            std::string metricStr = argv[++i];
            if (metricStr == "16bit" || metricStr == "16") {
                metricType = Metric::B16;
            } else if (metricStr == "32bit" || metricStr == "32") {
                metricType = Metric::B32;
            } else {
                std::cerr << "Error: Invalid metric type for " << arg << "." << std::endl;
                exit(1);
            }
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            std::string inputStr = argv[++i];
            if (inputStr == "HARD" || inputStr == "h") {
                inputType = ChannelIn::HARD;
            } else if (inputStr == "SOFT4" || inputStr == "s4") {   
                inputType = ChannelIn::SOFT4;
            } else if (inputStr == "SOFT8" || inputStr == "s8") {
                inputType = ChannelIn::SOFT8;
            } else if (inputStr == "SOFT16" || inputStr == "s16") {
                inputType = ChannelIn::SOFT16;
            } else if (inputStr == "FP32" || inputStr == "f") {
                inputType = ChannelIn::FP32;
            } else {
                std::cerr << "Error: Invalid input channel type for " << arg << "." << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Error: Unknown or incomplete argument: " << arg << std::endl;
            exit(1);
        }
    }
}