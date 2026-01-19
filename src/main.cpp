
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include "viterbiDF.h"

void parseArg(int argc, char *argv[], int& messageLen, float& snr, ChannelIn& inputType, Metric& metricType, DecodeOut& outputType, CompMode& compMode, bool& verbose);

template<ChannelIn inputType, Metric metricType, DecodeOut outputType, CompMode compMode>
void runPipeline(int messageLen, float snr, int& BENs, bool showStatus=false);
void runPipelineWrapper(ChannelIn inputType, Metric metricType, DecodeOut outputType, CompMode compMode, int messageLen, float snr, int& BENs, bool showStatus=false);

//-----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    // get parameters from command line
	ChannelIn inputType;
    Metric metricType;
    DecodeOut outputType;
    CompMode compMode;
    int messageLen;
	float snr;
    bool verbose;
    parseArg(argc, argv, messageLen, snr, inputType, metricType, outputType, compMode, verbose);
    if(metricType == Metric::M_B16 && inputType == ChannelIn::SOFT16){
        std::cerr << "Error: 16-bit metric does not support 16-bit soft decision input." << std::endl;
        return -1;
    }
    if(metricType == Metric::M_FP16 && inputType == ChannelIn::SOFT16){
        std::cerr << "Error: fp16 metric does not support 16-bit soft decision input." << std::endl;
        return -1;
    }
    if(metricType == Metric::M_FP16 && inputType == ChannelIn::SOFT8){
        std::cerr << "Error: fp16 metric does not support 8-bit soft decision input." << std::endl;
        return -1;
    }
    if(metricType == Metric::M_FP16 && compMode == CompMode::DPX){
        std::cerr << "Error: fp16 metric does not support DPX computation mode." << std::endl;
        return -1;
    }

    if(verbose){
        std::cout << "Message Length: " << messageLen << std::endl;
        std::cout << "SNR: " << snr << " dB" <<  std::endl;
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
        std::cout << "Metric Type: " << ((metricType == Metric::M_B16) ? "16-bit" : ((metricType == Metric::M_B32) ? "32-bit" : "FP16")) << std::endl;
        std::cout << "Output Type: " << ((outputType == DecodeOut::O_B16) ? "16-bit" : "32-bit") << std::endl;
        std::cout << "Computation Mode: " << ((compMode == CompMode::REG) ? "Regular" : "DPX") << std::endl;
        std::cout << std::endl;
    }

    // --- Dataflow Pipeline Execution Section ---
    int BENs; //bit error number
    double BERs; //bit error rate
    runPipelineWrapper(inputType, metricType, outputType, compMode, messageLen, snr, BENs, verbose);
	
	BERs = (double)BENs / messageLen;

    std::cout << "Pipeline executed." << std::endl;
    std::cout << "Final results -> BEN: " << BENs << "   BER: " << BERs << std::endl;
    // --- End of Pipeline Section ---
	
	
    return 0;
}

//-----------------------------------------------------------------------------

void runPipelineWrapper(ChannelIn inputType, Metric metricType, DecodeOut outputType, CompMode compMode, int messageLen, float snr, int& BENs, bool showStatus){
    if(metricType == Metric::M_B16){
        if(inputType == ChannelIn::HARD)           runPipeline<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT4)     runPipeline<ChannelIn::SOFT4, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT8)     runPipeline<ChannelIn::SOFT8, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>(messageLen, snr, BENs, showStatus);
        //--- never to be enabled ---// else if(inputType == ChannelIn::SOFT16)    runPipeline<ChannelIn::SOFT16, Metric::M_B16>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::FP32)      runPipeline<ChannelIn::FP32, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>(messageLen, snr, BENs, showStatus);
    }
    else if(metricType == Metric::M_B32){
        if(inputType == ChannelIn::HARD)           runPipeline<ChannelIn::HARD, Metric::M_B32, DecodeOut::O_B32, CompMode::REG>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT4)     runPipeline<ChannelIn::SOFT4, Metric::M_B32, DecodeOut::O_B32, CompMode::REG>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT8)     runPipeline<ChannelIn::SOFT8, Metric::M_B32, DecodeOut::O_B32, CompMode::REG>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT16)    runPipeline<ChannelIn::SOFT16, Metric::M_B32, DecodeOut::O_B32, CompMode::REG>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::FP32)      runPipeline<ChannelIn::FP32, Metric::M_B32, DecodeOut::O_B32, CompMode::REG>(messageLen, snr, BENs, showStatus);
    }
    else if(metricType == Metric::M_FP16){
        if(inputType == ChannelIn::HARD)           runPipeline<ChannelIn::HARD, Metric::M_FP16, DecodeOut::O_B16, CompMode::REG>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::SOFT4)     runPipeline<ChannelIn::SOFT4, Metric::M_FP16, DecodeOut::O_B16, CompMode::REG>(messageLen, snr, BENs, showStatus);
        //--- never to be enabled ---// else if(inputType == ChannelIn::SOFT8)     runPipeline<ChannelIn::SOFT8, Metric::M_FP16, DecodeOut::O_B16>(messageLen, snr, BENs, showStatus);
        //--- never to be enabled ---// else if(inputType == ChannelIn::SOFT16)    runPipeline<ChannelIn::SOFT16, Metric::M_FP16, DecodeOut::O_B16>(messageLen, snr, BENs, showStatus);
        else if(inputType == ChannelIn::FP32)      runPipeline<ChannelIn::FP32, Metric::M_FP16, DecodeOut::O_B16, CompMode::REG>(messageLen, snr, BENs, showStatus);
    }
    else{
        std::cerr << "Unsupported metric type." << std::endl;
    }
}

template<ChannelIn inputType, Metric metricType, DecodeOut outputType, CompMode compMode>
void runPipeline(int messageLen, float snr, int& BENs, bool showStatus){
    constexpr int CL = ViterbiCUDA<inputType, metricType, outputType, compMode>::constLen;
    constexpr int polyn1 = ViterbiCUDA<inputType, metricType, outputType, compMode>::polyn1;
    constexpr int polyn2 = ViterbiCUDA<inputType, metricType, outputType, compMode>::polyn2;
    constexpr int extraL = ViterbiCUDA<inputType, metricType, outputType, compMode>::extraL;
    constexpr int extraR = ViterbiCUDA<inputType, metricType, outputType, compMode>::extraR;
    using decVec_t = typename ViterbiDecoder<inputType, metricType, outputType, compMode>::decVec_t;

	int bitsPerPack = ViterbiDecoder<inputType, metricType, outputType, compMode>::bitsPerPack;
	
	// --- Dataflow Pipeline Setup ---
	std::random_device rd;
	// RandBitGen randGen(messageLen, 0);
	RandBitGen randGen (messageLen, rd());
    ConvolutionalEncoder convEnc(CL, polyn1, polyn2);
    AddNoise noise(pow(10, -snr/5.0), rd());
    // AddNoise noise(std::numeric_limits<float>::infinity());
    SoftDecisionPacker packer(inputType, 40000.0);
    ViterbiDecoder<inputType, metricType, outputType, compMode> viterbi;

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
            // printf("err ind = %d  decPack=0x%08x\n", i, decodedBitsVector[i/bitsPerPack]);
			// maxInd = i;
			// if(minInd == -1)
			// 	minInd = i;
		}
	}
    // printf("minInd: %d, maxInd: %d\n", minInd, maxInd);
    // --- End of BER Calculation Section ---
}

void parseArg(int argc, char *argv[], int& messageLen, float& snr, ChannelIn& inputType, Metric& metricType, DecodeOut& outputType, CompMode& compMode, bool& verbose) {
    // Set default values
    messageLen = 32000000;
    snr = 15.0;
    metricType = Metric::M_B32;
    inputType = ChannelIn::HARD;
    outputType = DecodeOut::O_B16;
    compMode = CompMode::REG;
    verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -n, --num <integer>      Set the message length.\n"
                      << "  -s, --snr <float>        Set the Signal-to-Noise Ratio (SNR).\n"
                      << "  -i, --input <type>       Set the input channel type (HARD|h, SOFT4|s4, SOFT8|s8, SOFT16|s16, FP32|f).\n"
                      << "  -m, --metric <type>      Set the metric type (b16, b32, f16).\n"
                      << "  -o, --output <type>      Set the output type (b16, b32).\n"
                      << "  -c, --compMode <type>    Set the computation mode (REG|reg, DPX|dpx).\n"
                      << "  -v, --verbose            Enable verbose output.\n"
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
            if (metricStr == "b16") {
                metricType = Metric::M_B16;
            } else if (metricStr == "b32") {
                metricType = Metric::M_B32;
            } else if (metricStr == "f16") {
                metricType = Metric::M_FP16;
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
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            std::string outputStr = argv[++i];
            if (outputStr == "b16") {
                outputType = DecodeOut::O_B16;
            } else if (outputStr == "b32") {
                outputType = DecodeOut::O_B32;
            } else {
                std::cerr << "Error: Invalid output type for " << arg << "." << std::endl;
                exit(1);
            }
        } else if ((arg == "-c" || arg == "--compMode") && i + 1 < argc) {
            std::string compModeStr = argv[++i];
            if (compModeStr == "REG" || compModeStr == "reg") {
                compMode = CompMode::REG;
            } else if (compModeStr == "DPX" || compModeStr == "dpx") {
                compMode = CompMode::DPX;
            } else {
                std::cerr << "Error: Invalid computation mode for " << arg << "." << std::endl;
                exit(1);
            }
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else {
            std::cerr << "Error: Unknown or incomplete argument: " << arg << std::endl;
            exit(1);
        }
    }
}