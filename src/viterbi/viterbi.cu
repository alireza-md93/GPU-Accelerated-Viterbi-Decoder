#include "cuda_runtime.h"
#include "gpuerrors.h"
#include "viterbi.h"
#include "viterbiBM.cuh"
#include "viterbiACS.cuh"
#include "viterbiTB.cuh"
#include "viterbiConsts.h"
#include <stdio.h>

template<int options>
struct ViterbiCUDA<options, true>::Impl {
	cudaEvent_t start; 
	cudaEvent_t stop;
};

template<int options>
ViterbiCUDA<options, true>::ViterbiCUDA()
: pathPrev_d(nullptr), dec_d(nullptr), enc_d(nullptr), preAllocated(false), blocksNum_total(16*400) 
{
	deviceSetup();
	pImpl = new Impl();
}

template<int options>
ViterbiCUDA<options, true>::ViterbiCUDA(size_t inputNum)
: pathPrev_d(nullptr), dec_d(nullptr), enc_d(nullptr), preAllocated(true)
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
	HANDLE_ERROR(cudaMalloc((void**)&enc_d, inputSize));
	HANDLE_ERROR(cudaMalloc((void**)&dec_d, outputSize));
	HANDLE_ERROR(cudaMalloc((void**)&pathPrev_d, ppSize));
}

template<int options>
void ViterbiCUDA<options, true>::memFree(){
	if(dec_d) 		{cudaFree(dec_d); dec_d = nullptr;}
	if(enc_d) 		{cudaFree(enc_d); enc_d = nullptr;}
	if(pathPrev_d)	{cudaFree(pathPrev_d); pathPrev_d = nullptr;}
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
{return (bmMemWidth * 4 * sizeof(metric_t) * blockDimY);}

template<int options>
size_t ViterbiCUDA<options, true>::getPathPrevSize()
{return (forwardLen / 8 * (1<<(constLen-1)) * blocksNum_total);}

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
template<ChannelIn inputType, Metric metricType, DecodeOut outputType, CompMode compMode>
__global__ void viterbi_core(decPack_t<outputType>* data, encPack_t<inputType>* coded, size_t messageLen, decPack_t<outputType>* pathPrev_all) {
	//coded: input coded array that contains 2*n bits with constraint mentioned above
	//data: output array that contains n allocated bits with constraint mentioned above
	
	extern __shared__ int sharedMem[];
	metric_t<metricType>* sharedMemTip = reinterpret_cast<metric_t<metricType>*>(sharedMem);
	sharedMemTip += ty * (bmMemWidth * 4);
	metric_t<metricType> (*branchMetric)[4] = (metric_t<metricType>(*)[4])sharedMemTip;

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
	trellisPM<metricType> oldPM, nowPM;
	trellisPP<outputType> oldPP, nowPP;
	bmCalcHelper<inputType> bmHelper;
	unsigned int allBmInd0, allBmInd1;
	bmIndCalc(allBmInd0, allBmInd1);
	int pmNormStride = 1 << (bpm<metricType> - chnWidth<inputType> - 2);
	/******************************************************************************************/

	int bmBatchLen = min(extraL + extraR, bmMemWidth);
	for(int bmBatch=0; bmBatch<extraL+extraR; bmBatch+=bmBatchLen){
		bmCalc<inputType, metricType>(bmBatch, bmBatchLen, branchMetric, coded, bmHelper);
		__syncwarp();
		for(int stage=bmBatch; stage<bmBatch+bmBatchLen; stage++)
			forwardACS<metricType, outputType>(stage, oldPM, oldPP, nowPM, nowPP, pathPrev, branchMetric, allBmInd0, allBmInd1, pmNormStride);
	}

	int slide;
	bmBatchLen = min(slideSize, bmMemWidth);
	for(slide=0; slide<decLen; slide+=slideSize){
		int stage = slide + extraL + extraR;
		for(int bmBatch=stage; bmBatch<stage+slideSize; bmBatch+=bmBatchLen){
			bmCalc<inputType, metricType>(bmBatch, bmBatchLen, branchMetric, coded, bmHelper);   
			__syncwarp();
			for(int i=bmBatch; i<bmBatch+bmBatchLen; i++){	
				forwardACS<metricType, outputType>(i, oldPM, oldPP, nowPM, nowPP, pathPrev, branchMetric, allBmInd0, allBmInd1, pmNormStride);
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
		forwardACS<metricType, outputType>(i, oldPM, oldPP, nowPM, nowPP, pathPrev, branchMetric, allBmInd0, allBmInd1, pmNormStride);
	}
	traceback<outputType>(stage+remSlideSize-1, slide+remSlideSize-1, remSlideSize, data, pathPrev);
}

//-----------------------------------------------------------------------------
template<int options>
void ViterbiCUDA<options, true>::run(encPack_t* input_h, decPack_t* output_h, size_t inputNum, float* kernelTime){
	size_t inputSize = getInputSize(inputNum);
	size_t messageLen = getMessageLen(inputNum);
	size_t outputSize = getOutputSize(inputNum);
	size_t sharedMemSize = getSharedMemSize();

	if(!preAllocated) memAlloc(inputNum);

	HANDLE_ERROR(cudaMemcpy(enc_d, input_h, inputSize, cudaMemcpyHostToDevice));
	
	dim3 grid (blocksNum_total/blockDimY, 1, 1); 
	dim3 block (32, blockDimY, 1);

	if(kernelTime){
		timerSetup();
		timerStart();
	}
	viterbi_core<inputType, metricType, outputType, compMode> <<<grid, block, sharedMemSize>>> (dec_d, enc_d, messageLen, pathPrev_d);
	if(kernelTime){
		timerStop();
		*kernelTime = timerElapsed();
	}
	HANDLE_ERROR(   cudaPeekAtLastError()   );

	HANDLE_ERROR(cudaMemcpy(output_h, dec_d, outputSize, cudaMemcpyDeviceToHost));

	if(!preAllocated) memFree();
}

#define INSTANTIATE_CASE(optionsFinal) template class ViterbiCUDA<optionsFinal>;

#define INSTANTIATE_COMP(optionsPrior) \
INSTANTIATE_CASE(optionsPrior | CompMode::REG) \
INSTANTIATE_CASE(optionsPrior | CompMode::DPX)

#define INSTANTIATE_DECODE(optionsPrior) \
INSTANTIATE_COMP(optionsPrior | DecodeOut::O_B16) \
INSTANTIATE_COMP(optionsPrior | DecodeOut::O_B32)

#define INSTANTIATE_METRIC(optionsPrior) \
INSTANTIATE_DECODE(optionsPrior | Metric::M_B16) \
INSTANTIATE_DECODE(optionsPrior | Metric::M_B32) \
INSTANTIATE_DECODE(optionsPrior | Metric::M_FP16)

#define INSTANTIATE_ALL \
INSTANTIATE_METRIC(ChannelIn::HARD) \
INSTANTIATE_METRIC(ChannelIn::SOFT4) 
INSTANTIATE_METRIC(ChannelIn::SOFT8) \
INSTANTIATE_METRIC(ChannelIn::SOFT16) \
INSTANTIATE_METRIC(ChannelIn::FP32)

INSTANTIATE_ALL

// template class ViterbiCUDA<ChannelIn::HARD | Metric::M_B16 | DecodeOut::O_B16 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::SOFT4 | Metric::M_B16 | DecodeOut::O_B16 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::SOFT8 | Metric::M_B16 | DecodeOut::O_B16 | CompMode::REG>;
// //--- never to be enabled ---// template class ViterbiCUDA<ChannelIn::SOFT16 | Metric::M_B16 | DecodeOut::O_B16 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::FP32 | Metric::M_B16 | DecodeOut::O_B16 | CompMode::REG>;

// template class ViterbiCUDA<ChannelIn::HARD | Metric::M_B32 | DecodeOut::O_B32 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::SOFT4 | Metric::M_B32 | DecodeOut::O_B32 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::SOFT8 | Metric::M_B32 | DecodeOut::O_B32 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::SOFT16 | Metric::M_B32 | DecodeOut::O_B32 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::FP32 | Metric::M_B32 | DecodeOut::O_B32 | CompMode::REG>;

// template class ViterbiCUDA<ChannelIn::HARD | Metric::M_FP16 | DecodeOut::O_B16 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::SOFT4 | Metric::M_FP16 | DecodeOut::O_B16 | CompMode::REG>;
// //--- never to be enabled ---// template class ViterbiCUDA<ChannelIn::SOFT8 | Metric::M_FP16 | DecodeOut::O_B16 | CompMode::REG>;
// //--- never to be enabled ---// template class ViterbiCUDA<ChannelIn::SOFT16 | Metric::M_FP16 | DecodeOut::O_B16 | CompMode::REG>;
// template class ViterbiCUDA<ChannelIn::FP32 | Metric::M_FP16 | DecodeOut::O_B16 | CompMode::REG>;