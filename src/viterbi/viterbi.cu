#include "cuda_runtime.h"
#include "gpuerrors.h"
#include "viterbi.h"
#include "viterbiBM.cuh"
#include "viterbiACS.cuh"
#include "viterbiTB.cuh"
#include "viterbiConsts.h"
#include <stdio.h>
#include <vector>
#include <algorithm>

template<int options>
struct ViterbiCUDA<options, true>::Impl {
	decPack_t* pathPrev_d;
	decPack_t* dec_d;
	encPack_t* enc_d;
	bool preAllocated;
	int blocksNum_total;
	cudaEvent_t start; 
	cudaEvent_t stop;
	Impl() : pathPrev_d(nullptr), dec_d(nullptr), enc_d(nullptr), preAllocated(false), blocksNum_total(16*400) {}
	~Impl() {}
};

template<int options>
ViterbiCUDA<options, true>::ViterbiCUDA()
: pImpl(new Impl())
{
	deviceSetup();
}

template<int options>
ViterbiCUDA<options, true>::ViterbiCUDA(size_t inputNum)
: pImpl(new Impl())
{
	deviceSetup();
	memAlloc(inputNum);
}

template<int options>
ViterbiCUDA<options, true>::~ViterbiCUDA(){
	memFree();
	delete pImpl;
}

template<int options>
void ViterbiCUDA<options, true>::memAlloc(size_t inputNum){
	size_t inputSize = getInputSize(inputNum);
	size_t outputSize = getOutputSize(inputNum);
	size_t ppSize = getPathPrevSize();
	
	//initialization
	HANDLE_ERROR(cudaMalloc((void**)&(pImpl->enc_d), inputSize));
	HANDLE_ERROR(cudaMalloc((void**)&(pImpl->dec_d), outputSize));
	HANDLE_ERROR(cudaMalloc((void**)&(pImpl->pathPrev_d), ppSize));
}

template<int options>
void ViterbiCUDA<options, true>::memFree(){
	if(pImpl->dec_d) 		{cudaFree(pImpl->dec_d); pImpl->dec_d = nullptr;}
	if(pImpl->enc_d) 		{cudaFree(pImpl->enc_d); pImpl->enc_d = nullptr;}
	if(pImpl->pathPrev_d)	{cudaFree(pImpl->pathPrev_d); pImpl->pathPrev_d = nullptr;}
}

template<int options>
size_t ViterbiCUDA<options, true>::getInputSize(size_t inputNum)
{
	if constexpr (inputType == ChannelIn::HARD) {
		return (roundup(inputNum, 8ULL) / 8);
	}
	else if constexpr (inputType == ChannelIn::SOFT4) {
		return (roundup(inputNum, 2ULL) / 2);
	}
	else if constexpr (inputType == ChannelIn::SOFT8) {
		return inputNum;
	}
	else if constexpr (inputType == ChannelIn::SOFT16) {
		return (inputNum * 2);	
	}
	else if constexpr (inputType == ChannelIn::FP32) {
		return (inputNum * sizeof(encPack_t));
	}
	else {
		return 0;
	}
}

template<int options>
size_t ViterbiCUDA<options, true>::getMessageLen(size_t inputNum)
{return ((inputNum / 2 - (extraL + extraR)) / bitsPerPack * bitsPerPack);}

template<int options>
size_t ViterbiCUDA<options, true>::getOutputSize(size_t inputNum)
{return (getMessageLen(inputNum)/8);}

template<int options>
size_t ViterbiCUDA<options, true>::getSharedMemSize()
{
	size_t shmemSize = bmMemWidth * 4 * sizeof(metric_t) * blockDimY;
	if constexpr (stateEx == StateExchng::SE_DIS) shmemSize += 64 * (sizeof(decPack_t) + sizeof(metric_t)) * blockDimY;
	return shmemSize;
}

template<int options>
size_t ViterbiCUDA<options, true>::getPathPrevSize()
{return (forwardLen / 8 * (1<<(constLen-1)) * pImpl->blocksNum_total);}

template<int options>
void ViterbiCUDA<options, true>::timerSetup(){
	cudaEventCreate(&(pImpl->start));
	cudaEventCreate(&(pImpl->stop));
}

template<int options>
void ViterbiCUDA<options, true>::timerStart(){
	cudaEventRecord(pImpl->start, 0);
}

template<int options>
void ViterbiCUDA<options, true>::timerStop(){
	cudaEventRecord(pImpl->stop, 0);
}

template<int options>
float ViterbiCUDA<options, true>::timerElapsed(){
	float elapsed;
	cudaEventSynchronize(pImpl->stop);
	cudaEventElapsedTime(&elapsed, pImpl->start, pImpl->stop);
	return elapsed;
}

template<int options>
void ViterbiCUDA<options, true>::timerDelete(){
	cudaEventDestroy(pImpl->start);
	cudaEventDestroy(pImpl->stop);
}

template<int options>
void ViterbiCUDA<options, true>::deviceSetup(){
	cudaSetDevice(0);
	// cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100ULL * 1024 * 1024);
	// cudaDeviceProp deviceProp;
	// HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
	// printf("Device %s has compute capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);
}

//-----------------------------------------------------------------------------
//the main core of viterbi decoder
//get data and polynoials ans decode 
template<ChannelIn inputType, Metric metricType, DecodeOut outputType, CompMode compMode, StateExchng stateEx>
__global__ void viterbi_core(decPack_t<outputType>* data, encPack_t<inputType>* coded, size_t messageLen, decPack_t<outputType>* pathPrev_all, size_t shmemSize_x) {
	//coded: input coded array that contains 2*n bits with constraint mentioned above
	//data: output array that contains n allocated bits with constraint mentioned above
	
	extern __shared__ char sharedMem[];
	char* sharedMemTip = sharedMem;
	sharedMemTip += ty * shmemSize_x;

	metric_t<metricType> (*branchMetric)[4] = (metric_t<metricType>(*)[4])sharedMemTip;
	sharedMemTip += bmMemWidth * 4 * sizeof(metric_t<metricType>);
	metric_t<metricType>* PM = (metric_t<metricType>*)sharedMemTip;
	sharedMemTip += 64 * sizeof(metric_t<metricType>);
	decPack_t<outputType>* PP = (decPack_t<outputType>*)sharedMemTip;
	sharedMemTip += 64 * sizeof(decPack_t<outputType>);

	decPack_t<outputType> (*pathPrev) [1<<(CL-1)] = (decPack_t<outputType> (*) [1<<(CL-1)])pathPrev_all + (bx*bdy + ty) * ((forwardLen-1)/bpp<outputType>+1);
	
	size_t packNum = messageLen / bpp<outputType>;
	size_t decLen = packNum / (gdx * bdy);
	size_t remPacks = packNum % (gdx * bdy);
	size_t startInd = decLen * (bx*bdy + ty) + min(remPacks, size_t(bx*bdy + ty));
	if( (bx*bdy + ty) < remPacks ) decLen++;
	decLen *= bpp<outputType>;
	startInd *= bpp<outputType>;
	
	data += startInd/bpp<outputType>;
	coded += startInd*2/dpp<inputType>;
	
	/****************************** calculate trellis parameters ******************************/	 
	bmCalcHelper<inputType> bmHelper;
	ForwardACS<metricType, outputType, compMode, stateEx> forwardACS;
	forwardACS.pmNormStride = 1 << (bpm<metricType> - chnWidth<inputType> - 2);
	forwardACS.branchMetric = branchMetric;
	if constexpr (stateEx == StateExchng::SE_EN){
		bmIndCalc(forwardACS.allBmInd0, forwardACS.allBmInd1);
	}
	else{
		forwardACS.bmOffset = __popc((tx<<1) & ViterbiCUDA<>::polyn1) % 2;
		forwardACS.bmOffset <<= 1;
		forwardACS.bmOffset += __popc((tx<<1) & ViterbiCUDA<>::polyn2) % 2;
		forwardACS.PM = PM;
		forwardACS.PP = PP;
		PM[tx] = 0;
		PM[tx+32] = 0;
	}
	/******************************************************************************************/

	int bmBatchLen = min(extraL + extraR, bmMemWidth);
	for(int bmBatch=0; bmBatch<extraL+extraR; bmBatch+=bmBatchLen){
		bmCalc<inputType, metricType>(bmBatch, bmBatchLen, branchMetric, coded, bmHelper);
		__syncwarp();
		for(int stage=bmBatch; stage<bmBatch+bmBatchLen; stage++)
			forwardACS.comp(stage, pathPrev);
	}

	int slide;
	bmBatchLen = min(slideSize, bmMemWidth);
	for(slide=0; slide<decLen; slide+=slideSize){
		int stage = slide + extraL + extraR;
		for(int bmBatch=stage; bmBatch<stage+slideSize; bmBatch+=bmBatchLen){
			bmCalc<inputType, metricType>(bmBatch, bmBatchLen, branchMetric, coded, bmHelper);   
			__syncwarp();
			for(int i=bmBatch; i<bmBatch+bmBatchLen; i++){	
				forwardACS.comp(i, pathPrev);
			}
		}
		__syncwarp();
		traceback<outputType>(stage+slideSize-1, slide+slideSize-1, slideSize, data, pathPrev);
	}

	int stage = slide + extraL + extraR;
	int remSlideSize = decLen % slideSize;
	bmCalc<inputType, metricType>(stage, remSlideSize, branchMetric, coded, bmHelper);   
	__syncwarp();
	for(int i=stage; i<stage+remSlideSize; i++){	
		forwardACS.comp(i, pathPrev);
	}
	traceback<outputType>(stage+remSlideSize-1, slide+remSlideSize-1, remSlideSize, data, pathPrev);
}

//-----------------------------------------------------------------------------
template<int options>
void ViterbiCUDA<options, true>::run(encPack_t* input_h, decPack_t* output_h, size_t inputNum, float* kernelTime){
	const float WARMUP = 500;
	const float ERROR = 0.05;
	const int MAX_TIMES = 100;
	size_t inputSize = getInputSize(inputNum);
	size_t messageLen = getMessageLen(inputNum);
	size_t outputSize = getOutputSize(inputNum);
	size_t sharedMemSize = getSharedMemSize();

	if(!(pImpl->preAllocated)) memAlloc(inputNum);

	HANDLE_ERROR(cudaMemcpy(pImpl->enc_d, input_h, inputSize, cudaMemcpyHostToDevice));
	
	dim3 grid (pImpl->blocksNum_total/blockDimY, 1, 1); 
	dim3 block (32, blockDimY, 1);

	std::vector<float> kernelTimeVec;
	
	if(kernelTime == nullptr){
		viterbi_core<inputType, metricType, outputType, compMode, stateEx> <<<grid, block, sharedMemSize>>> (pImpl->dec_d, pImpl->enc_d, messageLen, pImpl->pathPrev_d, sharedMemSize/blockDimY);
	}
	else{
		float warmupTime = 0.0;
		while(warmupTime < WARMUP && kernelTimeVec.size() < MAX_TIMES){
			timerSetup();
			timerStart();

			viterbi_core<inputType, metricType, outputType, compMode, stateEx> <<<grid, block, sharedMemSize>>> (pImpl->dec_d, pImpl->enc_d, messageLen, pImpl->pathPrev_d, sharedMemSize/blockDimY);

			timerStop();
			*kernelTime = timerElapsed();
			timerDelete();

			warmupTime += *kernelTime;
			kernelTimeVec.push_back(*kernelTime);
			// std::cout << *kernelTime << "," << warmupTime << std::endl;
		}
		std::sort(kernelTimeVec.begin(), kernelTimeVec.end());

		while((kernelTimeVec.back() - kernelTimeVec.front()) / kernelTimeVec.back() > ERROR){
			timerSetup();
			timerStart();

			viterbi_core<inputType, metricType, outputType, compMode, stateEx> <<<grid, block, sharedMemSize>>> (pImpl->dec_d, pImpl->enc_d, messageLen, pImpl->pathPrev_d, sharedMemSize/blockDimY);

			timerStop();
			*kernelTime = timerElapsed();
			timerDelete();

			auto it = std::lower_bound(kernelTimeVec.begin(), kernelTimeVec.end(), *kernelTime);
			kernelTimeVec.insert(it, *kernelTime);
			kernelTimeVec.pop_back();

			// std::cout << *kernelTime << std::endl;
			// for(auto t : kernelTimeVec) std::cout << t << ","; std::cout << std::endl;
			// std::cout << std::endl;
		}
	}

	HANDLE_ERROR(   cudaPeekAtLastError()   );

	HANDLE_ERROR(cudaMemcpy(output_h, pImpl->dec_d, outputSize, cudaMemcpyDeviceToHost));

	if(!(pImpl->preAllocated)) memFree();
}

#define INSTANTIATE_CASE(optionsFinal) template class ViterbiCUDA<optionsFinal>;

#define INSTANTIATE_DECODE(optionsPrior) \
INSTANTIATE_CASE(optionsPrior | DecodeOut::O_B16) \
INSTANTIATE_CASE(optionsPrior | DecodeOut::O_B32)

#define INSTANTIATE_INPUT(optionsPrior) \
INSTANTIATE_DECODE(optionsPrior | ChannelIn::HARD) \
INSTANTIATE_DECODE(optionsPrior | ChannelIn::SOFT4) \
INSTANTIATE_DECODE(optionsPrior | ChannelIn::SOFT8) \
INSTANTIATE_DECODE(optionsPrior | ChannelIn::SOFT16) \
INSTANTIATE_DECODE(optionsPrior | ChannelIn::FP32)

#define INSTANTIATE_COMP(optionsPrior) \
INSTANTIATE_INPUT(optionsPrior | CompMode::REG) \
INSTANTIATE_INPUT(optionsPrior | CompMode::DPX)


#define INSTANTIATE_METRIC(optionsPrior) \
INSTANTIATE_COMP(optionsPrior | Metric::M_B16) \
INSTANTIATE_COMP(optionsPrior | Metric::M_B32) \
INSTANTIATE_COMP(optionsPrior | Metric::M_FP16)

#define INSTANTIATE_ALL \
INSTANTIATE_METRIC(StateExchng::SE_EN) \
INSTANTIATE_METRIC(StateExchng::SE_DIS)

INSTANTIATE_ALL