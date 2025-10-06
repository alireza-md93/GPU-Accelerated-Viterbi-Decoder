#pragma once

#include "parameters.h"

int viterbi_run(float* input_d, path_t* output_d, int messageLen, float* time, ACS acsType);
