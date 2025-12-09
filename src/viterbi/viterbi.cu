#include "cuda_runtime.h"
// #include "gputimer.h"
#include "gpuerrors.h"
#include "viterbi.h"
#include "viterbiBM.cuh"
#include "viterbiACS.cuh"
#include "viterbiTB.cuh"
#include "viterbiConsts.h"
#include <stdio.h>

template<Metric metricType, ChannelIn inputType>
struct ViterbiCUDA<metricType, inputType>::Impl {
	cudaEvent_t start; 
	cudaEvent_t stop;
};

template<Metric metricType, ChannelIn inputType>
ViterbiCUDA<metricType, inputType>::ViterbiCUDA()
: pathPrev_d(nullptr), dec_d(nullptr), enc_d(nullptr), preAllocated(false), blocksNum_total(16*400) 
{
	deviceSetup();
	pImpl = new Impl();
}

template<Metric metricType, ChannelIn inputType>
ViterbiCUDA<metricType, inputType>::ViterbiCUDA(size_t inputNum)
: pathPrev_d(nullptr), dec_d(nullptr), enc_d(nullptr), preAllocated(true)
{
	deviceSetup();
	memAlloc(inputNum);
}

template<Metric metricType, ChannelIn inputType>
ViterbiCUDA<metricType, inputType>::~ViterbiCUDA(){
	memFree();
	delete pImpl;
}

template<Metric metricType, ChannelIn inputType>
void ViterbiCUDA<metricType, inputType>::memAlloc(size_t inputNum){
	size_t inputSize = getInputSize(inputNum);
	size_t outputSize = getOutputSize(inputNum);
	size_t ppSize = getPathPrevSize();
	
	//initialization
	HANDLE_ERROR(cudaMalloc((void**)&enc_d, inputSize));
	HANDLE_ERROR(cudaMalloc((void**)&dec_d, outputSize));
	HANDLE_ERROR(cudaMalloc((void**)&pathPrev_d, ppSize));
}

template<Metric metricType, ChannelIn inputType>
void ViterbiCUDA<metricType, inputType>::memFree(){
	if(dec_d) 		{cudaFree(dec_d); dec_d = nullptr;}
	if(enc_d) 		{cudaFree(enc_d); enc_d = nullptr;}
	if(pathPrev_d)	{cudaFree(pathPrev_d); pathPrev_d = nullptr;}
}

template<Metric metricType, ChannelIn inputType>
size_t ViterbiCUDA<metricType, inputType>::getInputSize(size_t inputNum)
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

template<Metric metricType, ChannelIn inputType>
size_t ViterbiCUDA<metricType, inputType>::getMessageLen(size_t inputNum)
{return ((inputNum / 2 - (extraL + extraR)) / bitsPerPack * bitsPerPack);}

template<Metric metricType, ChannelIn inputType>
size_t ViterbiCUDA<metricType, inputType>::getOutputSize(size_t inputNum)
{return (getMessageLen(inputNum)/8);}

template<Metric metricType, ChannelIn inputType>
size_t ViterbiCUDA<metricType, inputType>::getSharedMemSize()
{return (shmemWidth * 4 * sizeof(metric_t) * blockDimY);}

template<Metric metricType, ChannelIn inputType>
size_t ViterbiCUDA<metricType, inputType>::getPathPrevSize()
{return (shmemWidth / 8 * (1<<(constLen-1)) * blocksNum_total);}

template<Metric metricType, ChannelIn inputType>
void ViterbiCUDA<metricType, inputType>::timerSetup(){
	cudaEventCreate(&(pImpl->start));
	cudaEventCreate(&(pImpl->stop));
}

template<Metric metricType, ChannelIn inputType>
void ViterbiCUDA<metricType, inputType>::timerStart(){
	cudaEventRecord(pImpl->start, 0);
}

template<Metric metricType, ChannelIn inputType>
void ViterbiCUDA<metricType, inputType>::timerStop(){
	cudaEventRecord(pImpl->stop, 0);
}

template<Metric metricType, ChannelIn inputType>
float ViterbiCUDA<metricType, inputType>::timerElapsed(){
	float elapsed;
	cudaEventSynchronize(pImpl->stop);
	cudaEventElapsedTime(&elapsed, pImpl->start, pImpl->stop);
	return elapsed;
}

template<Metric metricType, ChannelIn inputType>
void ViterbiCUDA<metricType, inputType>::timerDelete(){
	cudaEventDestroy(pImpl->start);
	cudaEventDestroy(pImpl->stop);
}

template<Metric metricType, ChannelIn inputType>
void ViterbiCUDA<metricType, inputType>::deviceSetup(){
	cudaSetDevice(0);
	// cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100ULL * 1024 * 1024);
	// cudaDeviceProp deviceProp;
	// HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
	// printf("Device %s has compute capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);
}

//-----------------------------------------------------------------------------
//the main core of viterbi decoder
//get data and polynoials ans decode 
template<Metric metricType, ChannelIn inputType>
__global__ void viterbi_core(decPack_t<metricType>* data, encPack_t<inputType>* coded, size_t messageLen, decPack_t<metricType>* pathPrev_all) {
	//coded: input coded array that contains 2*n bits with constraint mentioned above
	//data: output array that contains n allocated bits with constraint mentioned above
	
	extern __shared__ int sharedMem[];
	metric_t<metricType>* sharedMemTip = reinterpret_cast<metric_t<metricType>*>(sharedMem);
	sharedMemTip += ty * (shmemWidth * 4);
	metric_t<metricType> (*branchMetric)[4] = (metric_t<metricType>(*)[4])sharedMemTip;

	decPack_t<metricType> (*pathPrev) [1<<(CL-1)] = (decPack_t<metricType> (*) [1<<(CL-1)])pathPrev_all + (bx*bdy + ty) * ((shmemWidth-1)/bpp<metricType>+1);
	
	size_t packNum = messageLen / bpp<metricType>;
	size_t decLen = packNum / (gdx * bdy);
	size_t remPacks = packNum % (gdx * bdy);
	size_t startInd = decLen * (bx*bdy + ty) + min(remPacks, size_t(bx*bdy + ty));
	if( (bx*bdy + ty) < remPacks ) decLen++;
	decLen *= bpp<metricType>;
	startInd *= bpp<metricType>;
	
	data += startInd/bpp<metricType>;
	coded += startInd*2/dpp<inputType>;
	
	/****************************** calculate trellis parameters ******************************/	 
	trellis<metricType> old, now;
	bmCalcHelper<inputType> bmHelper;
	unsigned int allBmInd0, allBmInd1;
	bmIndCalc(allBmInd0, allBmInd1);
	int pmNormStride = 1 << (bpm<metricType> - chnWidth<inputType> - 2);
	/******************************************************************************************/

	bmCalc<metricType, inputType>(0, extraL+extraR, branchMetric, coded, bmHelper);
	__syncwarp();
	for(int stage=0; stage<extraL+extraR; stage++)
		forwardACS<metricType>(stage, old, now, pathPrev, branchMetric, allBmInd0, allBmInd1, pmNormStride);

	int slide;
	for(slide=0; slide<decLen; slide+=slideSize){
		int stage = slide + extraL + extraR;
		bmCalc<metricType, inputType>(stage, slideSize, branchMetric, coded, bmHelper);   
		__syncwarp();
		for(int i=stage; i<stage+slideSize; i++){	
			forwardACS<metricType>(i, old, now, pathPrev, branchMetric, allBmInd0, allBmInd1, pmNormStride);
		}
		__syncwarp();
		traceback<metricType>(stage+slideSize-1, slide+slideSize-1, slideSize, data, pathPrev);
	}

	int stage = slide + extraL + extraR;
	int remSlideSize = decLen % slideSize;
	bmCalc<metricType, inputType>(stage, remSlideSize, branchMetric, coded, bmHelper);   
	__syncwarp();
	for(int i=stage; i<stage+remSlideSize; i++){	
		forwardACS<metricType>(i, old, now, pathPrev, branchMetric, allBmInd0, allBmInd1, pmNormStride);
	}
	traceback<metricType>(stage+remSlideSize-1, slide+remSlideSize-1, remSlideSize, data, pathPrev);
}

//-----------------------------------------------------------------------------
template<Metric metricType, ChannelIn inputType>
void ViterbiCUDA<metricType, inputType>::run(encPack_t* input_h, decPack_t* output_h, size_t inputNum, float* kernelTime){
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
	viterbi_core<metricType, inputType> <<<grid, block, sharedMemSize>>> (dec_d, enc_d, messageLen, pathPrev_d);
	if(kernelTime){
		timerStop();
		*kernelTime = timerElapsed();
	}
	HANDLE_ERROR(   cudaPeekAtLastError()   );

	HANDLE_ERROR(cudaMemcpy(output_h, dec_d, outputSize, cudaMemcpyDeviceToHost));

	if(!preAllocated) memFree();
}

template class ViterbiCUDA<Metric::B16, ChannelIn::HARD>;
template class ViterbiCUDA<Metric::B16, ChannelIn::SOFT4>;
template class ViterbiCUDA<Metric::B16, ChannelIn::SOFT8>;
//--- never to be enabled ---// template class ViterbiCUDA<Metric::B16, ChannelIn::SOFT16>;
template class ViterbiCUDA<Metric::B16, ChannelIn::FP32>;

template class ViterbiCUDA<Metric::B32, ChannelIn::HARD>;
template class ViterbiCUDA<Metric::B32, ChannelIn::SOFT4>;
template class ViterbiCUDA<Metric::B32, ChannelIn::SOFT8>;
template class ViterbiCUDA<Metric::B32, ChannelIn::SOFT16>;
template class ViterbiCUDA<Metric::B32, ChannelIn::FP32>;

template class ViterbiCUDA<Metric::FP16, ChannelIn::HARD>;
template class ViterbiCUDA<Metric::FP16, ChannelIn::SOFT4>;
//--- never to be enabled ---// template class ViterbiCUDA<Metric::FP16, ChannelIn::SOFT8>;
//--- never to be enabled ---// template class ViterbiCUDA<Metric::FP16, ChannelIn::SOFT16>;
template class ViterbiCUDA<Metric::FP16, ChannelIn::FP32>;