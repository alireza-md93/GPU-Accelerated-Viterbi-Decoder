#pragma once

#include "parameters.h"

template<Metric metricType>
void viterbi_run(float* input_d, pack_t<metricType>* output_d, int messageLen, float* time);
