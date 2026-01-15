#include "viterbi.h"
#include "viterbiConsts.h"

template<Metric metricType>
__device__ void traceback(int endStage, int dataEndInd, int tbLength, decPack_t<metricType>* data, decPack_t<metricType> pathPrev[][1<<(CL-1)]){
	if(tx == 0){
		int winnerState = 0;
	
		for(int s=0; s<extraR-bpp<metricType>; s+=bpp<metricType>){
			decPack_t<metricType> pp = pathPrev[((endStage - s) % forwardLen)/bpp<metricType>][winnerState];
			winnerState = __brev(pp<<(32-bpp<metricType>)) & ((1U<<(CL-1))-1);
		}
		
		for(int s=0; s<tbLength; s+=bpp<metricType>){
			int i = (endStage - extraR - s) % forwardLen;
			decPack_t<metricType> pp = pathPrev[i/bpp<metricType>][winnerState];
			data[(dataEndInd-s)/bpp<metricType>] = pp;
			winnerState = __brev(pp<<(32-bpp<metricType>)) & ((1U<<(CL-1))-1);
		}
	}
}