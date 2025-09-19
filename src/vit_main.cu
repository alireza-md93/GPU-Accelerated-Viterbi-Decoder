
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

#include "viterbi.h"
#include "gpuerrors.h"
#include "gputimer.h"

//=====================================> GPU parameters <====================================//
#define SHFTL_RAW 32 //shift_left:the length of left overlapping part of this block with the lefthand block (without puncturing)
#define SHFTR_RAW 32 //shift_right:the length of right overlapping part of this block with the righthand block (without puncturing)
#define D_RAW 256 //the length of the valid output data of each block (without puncturing)
#define WINSIZE (DECSIZE + SHFTL + SHFTR) //total size of each block
//===========================================================================================//



//=====================================> coding parameters <====================================//
#define CL 7 //constraint length
#define POLYN1 0171 //polynomial 1
#define POLYN2 0133 //polynomial 2

#define TPB (1<<(CL-2))

#define PI 3.141592653589793
//==============================================================================================//

#define PATHSIZE 8
typedef unsigned char path_t;

// #define PATHSIZE 32
// typedef unsigned int path_t;

#define ROUNDUP(a, b) ( (a)<=0 ? 0 : (  ( (((a)-1)/(b)) + 1 ) * (b) ) ) //round up a to the nearest multiple of b

//=====================================> GPU parameters <====================================//
#define SHFTL ROUNDUP(SHFTL_RAW, PATHSIZE)
#define SHFTR ROUNDUP(SHFTR_RAW, PATHSIZE)
#define DECSIZE  ROUNDUP(D_RAW, PATHSIZE)
//===========================================================================================//


// ===========================> Functions Prototype <===============================
void noiseN(bool data[], float noisy[], int num, float std);
void fill(bool* data, int size);
void get_inputs(int argc, char *argv[], int& messageLen, float& snr);
void gpuKernels(bool* data, float* coded, int messageLen, float* gpuKernelTime);
void convEnc(bool data[], bool coded[], int n);
// =================================================================================

int main(int argc, char *argv[]) {

    // get parameters from command line
    int messageLen;
	float snr;
    get_inputs(argc, argv, messageLen, snr);
	int messageLen_ext = messageLen + SHFTL + SHFTR;
	int codedLen = 2*messageLen_ext;
	
	int deviceNumbers;
	cudaGetDeviceCount(&deviceNumbers);
	std::cout << "Number of CUDA devices: " << deviceNumbers << std::endl;
	cudaSetDevice(deviceNumbers-1);
	
	int i;
	 
	int BENs;//bit error number; 
	double BERs;//bit error rate
	
	//memory in CPU side
	bool* data;
	bool* dataDec_SH; //SH: soft input hard output
	bool* coded;
	float* codedNoisy;
	
	//allocate memory for CPU
	data = (bool*) malloc(messageLen * sizeof(bool));
	dataDec_SH = (bool*) malloc(messageLen * sizeof(bool));
	coded = (bool*) malloc(codedLen * sizeof(bool));
	codedNoisy = (float*) malloc(codedLen * sizeof(float));

	BENs = 0;
	
	// fill data
	fill(data, messageLen_ext);
	
	//encode
	convEnc(data, coded, messageLen_ext);
	
	//add soft noise
	float std = pow(10, -snr/20.0);
	noiseN(coded, codedNoisy, codedLen, std);

	// for(int i=0; i<codedLen; i++)
	// 	if(coded[i])
	// 		codedNoisy[i] = 1.0;
	// 	else
	// 		codedNoisy[i] = -1.0;

	
	//soft decode
	float gpuKernelTime;
	clock_t t2 = clock();
	
	gpuKernels(dataDec_SH, codedNoisy, messageLen, &gpuKernelTime);
	
	clock_t t3 = clock();
	float t_elp = (float)(t3-t2)/(CLOCKS_PER_SEC/1000); //the time of kernel and data transfer
	
	//check
	int minInd=-1; //minimum index of errors
	int maxInd = 0; //maximum index of errors
	for(i=SHFTL; i<messageLen_ext-SHFTR; i++){
		if(dataDec_SH[i-SHFTL] != data[i]){
			//printf(":%d\n", i);
			BENs++;
			maxInd = i;
			if(minInd == -1)
				minInd = i;
		}
	}
	printf("min:%d\tmax:%d\n", minInd, maxInd);
	
	BERs = (double)BENs / messageLen;
	
	printf("snr:%f\tBEN:%d   \tBER:%f\ttime:%f\ttime_ker:%f\n", snr, BENs, BERs, t_elp, gpuKernelTime);
	
    // free allocated memory for later use
	free(data);
    free(dataDec_SH);
    free(coded);
    free(codedNoisy);
	
    return 0;
}

//-----------------------------------------------------------------------------
//generate normal distribution noise with a user defined std and add it to input
//data is input array which is boolean. its true elements are considered 1 and its
//false ones are considered -1 and then generated noise will be added to make a
//float array as an output named noisy.
void noiseN(bool data[], float noisy[], int num, float std)
{
  float r1;
  float r2;
  float x;
  int max = RAND_MAX/2;
  
  int i; for (i=0; i<num; i++) {
  	r1 = (float)(rand()%max+1) / (max+1);
	r2 = (float)(rand()%max+1) / (max+1);
	x = sqrt ( - 2.0 * log ( r1 ) ) * cos ( 2.0 * PI * r2 );
  	
    if(data[i]){
    	noisy[i] = (float)(1.0 + std*x);
	}
	else{
		noisy[i] = (float)(-1.0 + std*x);
	}
	
  }

}
//-----------------------------------------------------------------------------
//encoded bits (enc) in host is received and decoded bits (dec) in host will be provided
//all GPU memory allocation and data transfer between host and device is done in the function.
//n is the numer of decoded bits.
//wrap is flag that defines whether to partition data or not
//gpuKernelTime will be filled the time of kernel without data transfer
void gpuKernels(bool* dec, float* enc, int messageLen, float* gpuKernelTime) {
    bool* dec_d;
	//bool* coded_trel_d;
    float* enc_d;
	
	int inputSize = (messageLen + SHFTL + SHFTR)*2 * sizeof(float);
	int outputSize = messageLen * sizeof(bool);
	
    //HANDLE_ERROR(cudaMemcpy(coded_d, coded, inputSize, cudaMemcpyHostToDevice));
	//initialization
	HANDLE_ERROR(cudaMalloc((void**)&dec_d, outputSize));
	HANDLE_ERROR(cudaMalloc((void**)&enc_d, inputSize));
	
	//run
	HANDLE_ERROR(cudaMemcpy(enc_d, enc, inputSize, cudaMemcpyHostToDevice));
	viterbi_run (enc_d, dec_d, messageLen, gpuKernelTime);
	HANDLE_ERROR(cudaMemcpy(dec, dec_d, outputSize, cudaMemcpyDeviceToHost));
	
	
    HANDLE_ERROR(cudaFree(dec_d));
	//HANDLE_ERROR(cudaFree(coded_trel_d));
    HANDLE_ERROR(cudaFree(enc_d));
	
}
//-----------------------------------------------------------------------------
//gets argc and argv and generates initial parameters
void get_inputs(int argc, char *argv[], int& messageLen, float& snr)
{
	int temp;
	switch(argc){
		case 1:
			messageLen = 32000000 - (32000000 % DECSIZE);
			snr = 15.0;
			break;
			
		case 2:
			temp = atoi(argv[1]);
			messageLen = temp - (temp % DECSIZE);
			snr = 15.0;
			break;
			
		case 3:
			temp = atoi(argv[1]);
			messageLen = temp - (temp % DECSIZE);
			snr = atof(argv[4]);
			break;
			
		default:
			printf("more than required inputs!!!\n");
			exit(1);
	}
}
//-----------------------------------------------------------------------------
//fills boolean data with true and false with the same probability
void fill(bool* data, int size) {
    srand( time(NULL) );
	int i; for(i=0; i<size; i++){
		data[i] = ( rand() > (RAND_MAX/2) );
		//data[i] = true;
	}
}
//-----------------------------------------------------------------------------
//convolutional encoder
//data will be encoded to coded
//n is the size of data
//r is the decimator of coding rate fraction
//k is constraint length
//polyn is the array of polynomials
void convEnc(bool data[], bool coded[], int n){
	int temp;
	unsigned int buffer = 0;
	int polyn[2] = {POLYN1, POLYN2};
		
	for(int i=0; i<n; i++){
		
		buffer >>= 1;
		buffer |= ((int)(data[i]) << (CL-1));
		
		for(int j=0; j<2; j++){
			temp = buffer & polyn[j];
			coded[2*i + j] = 0;
			for(int cnt=0; cnt<CL; cnt++){
				coded[2*i + j] ^= (temp & 1);
				temp >>= 1;
			}
			
		}	
		
	}
}