#ifndef RADIXSORT_H
#define RADIXSORT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>


extern "C" void radixSort_Sample(int NSize, double* Host_TestArrayIn, double* Host_TestArrayOut, int dir, float & timerRadixSort);

#endif



