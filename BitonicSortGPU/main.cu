#include "bitnicSort.h"
#include "radixSort.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <math.h>
#include <algorithm>
#include "MyNeighborList.h"
#include "MyNeighborList_multipleBox.h"

extern "C" int ComparetGT(const void *ValA,const void *ValB) {

	//std::cout <<"GT "<< (*(double*)ValA >= *(double*)ValB) << std::endl;

	return (*(double*)ValA > *(double*)ValB)? 1: -1;
}

extern "C" int ComparetLT(const void *ValA, const void *ValB) {

	//std::cout << "LT " << (*(double*)ValA <= *(double*)ValB) << std::endl;

	return (*(double*)ValB > *(double*)ValA)? 1: -1;
}


int main_wait(int argc, char** argv) {
	int NSize;
	double *Host_TestArrayIn;
	double *Host_TestArrayOut;
	double *Host_TestArrayOut_Radix;
	int currentDevice;
	int noone;
	float timer;
	float timerRadixSort;

	timer = 0.E0;

	timerRadixSort = 0.E0;

	int err = cudaGetDevice(&currentDevice);

	err = cudaSetDevice(0);

	srand(888);

	for (int NRand = 0; NRand < 1; NRand++) {

		for (NSize = 10; NSize <= 5000; NSize++) {

			Host_TestArrayIn = new double[NSize];
			Host_TestArrayOut = new double[NSize];

			Host_TestArrayOut_Radix = new double[NSize];

			for (int i = 0; i < NSize; i++) {
				Host_TestArrayIn[i] = rand();
			}

			std::cout << "*************************************" << std::endl;
			for (int j = 0; j < NSize; j++) {
				std::cout << Host_TestArrayIn[j] << " ";
			}
			std::cout << std::endl;

			ArbitraryBitonicSort(NSize, Host_TestArrayIn, Host_TestArrayOut, 1, timer);

			radixSort_Sample(NSize, Host_TestArrayIn, Host_TestArrayOut_Radix, 1, timerRadixSort);

			std::qsort(Host_TestArrayIn, NSize, sizeof(double), ComparetGT);

			//if(26 == NSize){
			std::cout << "*******Check for size: " << NSize << std::endl;
			for (int i = 0; i < NSize; i++) {
				
				if (Host_TestArrayIn[i] != Host_TestArrayOut[i]) {
					std::cout << "Wrong for arbitrary bitinic sorting " << Host_TestArrayIn[i] << " for array size is " << NSize << std::endl;
					std::cout << "The position is " << i << std::endl;

					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayOut[j] << " ";
					}
					std::cout << std::endl;


					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayIn[j] << " ";
					}
					std::cout << std::endl;

					std::cin >> noone;
				}


				if (Host_TestArrayIn[i] != Host_TestArrayOut_Radix[i]) {
					std::cout << "Wrong for radix sorting " << Host_TestArrayIn[i] << " for array size is " << NSize << std::endl;
					std::cout << "The position is " << i << std::endl;

					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayOut_Radix[j] << " ";
					}
					std::cout << std::endl;


					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayIn[j] << " ";
					}
					std::cout << std::endl;

					std::cin >> noone;
				}








			}

			//}

			delete[] Host_TestArrayIn;
			delete[] Host_TestArrayOut;
			delete[] Host_TestArrayOut_Radix;
		}

	}

	err = cudaSetDevice(currentDevice);

	return 0;
}


int main33(int argc, char** argv) {
	int NSize;
	double *Host_TestArrayIn;
	double *Host_TestArrayOut;
	double *Host_TestArrayOut_Arb;
	double *Host_TestArrayOut_Radix;
	double *Dev_TestArrayIn;
	double *Dev_TestArrayOut;
	int currentDevice;
	int noone;
	float timerBitinicSort;
	float totalTimerBitinicSort;
	float timerRadixSort;
	float totalTimerRadixSort;
	float timerThrust;
	float totalTimerThrust;
	float timerArbitraryBitonicSort;
	float totalArbitraryBitonicSort;
	float timerCPU;
	float totalTimerCPU;
	thrust::host_vector<double> Host_thrust;
	thrust::device_vector<double> Device_thrust;

	int err = cudaGetDevice(&currentDevice);

	cudaEvent_t StartEventBitinicSort;
	cudaEvent_t StopEventBitinicSort;

	cudaEvent_t StartEventThrust;
	cudaEvent_t StopEventThrust;

	cudaEvent_t StartEventCPU;
	cudaEvent_t StopEventCPU;

	err = cudaSetDevice(0);

	srand(55352);

	totalTimerBitinicSort = 0.E0;

	totalTimerThrust = 0.E0;

	totalArbitraryBitonicSort = 0.E0;

	timerArbitraryBitonicSort = 0.E0;

	totalTimerRadixSort = 0.E0;

	timerRadixSort = 0.E0;


	totalTimerCPU = 0.E0;

	cudaEventCreate(&StartEventBitinicSort, 0);
	cudaEventCreate(&StopEventBitinicSort, 0);

	cudaEventCreate(&StartEventThrust, 0);
	cudaEventCreate(&StopEventThrust, 0);

	cudaEventCreate(&StartEventCPU, 0);
	cudaEventCreate(&StopEventCPU, 0);

	for (int NRand = 0; NRand < 1; NRand++) {
		int Nbox = 800;
		for (NSize = 50* Nbox; NSize < 100000000; NSize<<=1) {

			Host_TestArrayIn = new double[NSize];
			Host_TestArrayOut = new double[NSize];


			for (int i = 0; i < NSize; i++) {
				Host_TestArrayIn[i] = double(rand())/RAND_MAX;
				Host_thrust.push_back(Host_TestArrayIn[i]);
			}

			/*
			std::cout << "*************************************" << std::endl;
			for (int j = 0; j < NSize; j++) {
				std::cout << Host_TestArrayIn[j] << " ";
			}
			std::cout << std::endl;
			*/

			int error = cudaMalloc((void**)&Dev_TestArrayIn, NSize * sizeof(double));

			error = cudaMalloc((void**)&Dev_TestArrayOut, NSize * sizeof(double));

			cudaMemcpy(Dev_TestArrayIn, Host_TestArrayIn, NSize * sizeof(double), cudaMemcpyHostToDevice);

			cudaEventRecord(StartEventBitinicSort, 0);

			//bitonicSort(Dev_TestArrayOut, Dev_TestArrayIn,1,NSize,1);

			cudaEventRecord(StopEventBitinicSort, 0);

			cudaEventSynchronize(StopEventBitinicSort);

			cudaEventElapsedTime(&timerBitinicSort, StartEventBitinicSort,StopEventBitinicSort);

			totalTimerBitinicSort = totalTimerBitinicSort + timerBitinicSort;

			std::cout << "The elapse time is (ms): " <<std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerBitinicSort << " for BitinicSort size: " << NSize << std::endl;

			cudaMemcpy(Host_TestArrayOut, Dev_TestArrayOut, NSize * sizeof(double), cudaMemcpyDeviceToHost);


			/*ArbitraryBitonicSort*/
			Host_TestArrayOut_Arb = new double[NSize];

			ArbitraryBitonicSort(NSize, Host_TestArrayIn, Host_TestArrayOut_Arb, 1, timerArbitraryBitonicSort);

			totalArbitraryBitonicSort = totalArbitraryBitonicSort + timerArbitraryBitonicSort;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerArbitraryBitonicSort << " for Arbitrary bitonic sort size: " << NSize << std::endl;


			/*Radix Sort*/
			Host_TestArrayOut_Radix = new double[NSize];

			radixSort_Sample(NSize, Host_TestArrayIn, Host_TestArrayOut_Radix, 1, timerRadixSort);

			totalTimerRadixSort = totalTimerRadixSort + timerRadixSort;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerRadixSort << " for radix sort size: " << NSize << std::endl;

			/*Thrust*/
			Device_thrust = Host_thrust;

			cudaEventRecord(StartEventThrust, 0);

			thrust::sort(Device_thrust.begin(), Device_thrust.end());

			cudaEventRecord(StopEventThrust, 0);

			cudaEventSynchronize(StopEventThrust);

			cudaEventElapsedTime(&timerThrust, StartEventThrust, StopEventThrust);

			totalTimerThrust = totalTimerThrust + timerThrust;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerThrust << " for thrust sort size: " << NSize << std::endl;

			thrust::copy(Device_thrust.begin(), Device_thrust.end(), Host_thrust.begin());

			/*CPU*/
			cudaEventRecord(StartEventCPU, 0);

			clock_t timeStart = clock();

			std::qsort(Host_TestArrayIn, NSize, sizeof(double), ComparetGT);

			clock_t timeEnd = clock();

			cudaEventRecord(StopEventCPU, 0);

			cudaEventSynchronize(StopEventCPU);

			cudaEventElapsedTime(&timerCPU, StartEventCPU, StopEventCPU);

			totalTimerCPU = totalTimerCPU + timeEnd - timeStart;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timeEnd - timeStart << " for cpu sort size: " << NSize << std::endl;

			std::cout << "*******Check for size: " << NSize << std::endl;
			for (int i = 0; i < NSize; i++) {
				
				/*
				if (Host_TestArrayIn[i] != Host_TestArrayOut[i]) {
					std::cout << "Wrong for bitinic sorting " << Host_TestArrayIn[i] << " for array size is " << NSize << std::endl;
					std::cout << "The position is " << i << std::endl;

					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayOut[j] << " ";
					}
					std::cout << std::endl;


					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayIn[j] << " ";
					}
					std::cout << std::endl;

					std::cin >> noone;
				}*/
				
				
				if (Host_TestArrayIn[i] != Host_TestArrayOut_Arb[i]) {
					std::cout << "Wrong for arbitrary bitinic sorting " << Host_TestArrayIn[i] << " for array size is " << NSize << std::endl;
					std::cout << "The position is " << i << std::endl;

					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayOut_Arb[j] << " ";
					}
					std::cout << std::endl;


					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayIn[j] << " ";
					}
					std::cout << std::endl;

					std::cin >> noone;
				}

				/*
				if (Host_TestArrayIn[i] != Host_TestArrayOut_Radix[i]) {
					std::cout << "Wrong for radix sorting " << Host_TestArrayIn[i] << " for array size is " << NSize << std::endl;
					std::cout << "The position is " << i << std::endl;

					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayOut_Radix[j] << " ";
					}
					std::cout << std::endl;


					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayIn[j] << " ";
					}
					std::cout << std::endl;

					std::cin >> noone;
				}
				*/

				if (Host_TestArrayIn[i] != Host_thrust[i]) {
					std::cout << "Wrong for thrust sorting " << Host_TestArrayIn[i] << " for array size is " << NSize << std::endl;
					std::cout << "The position is " << i << std::endl;

					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_thrust[j] << " ";
					}
					std::cout << std::endl;


					std::cout << "*************************************" << std::endl;
					for (int j = 0; j < NSize; j++) {
						std::cout << Host_TestArrayIn[j] << " ";
					}
					std::cout << std::endl;

					std::cin >> noone;
				}
			}

			delete[] Host_TestArrayIn;
			delete[] Host_TestArrayOut;
			delete[] Host_TestArrayOut_Arb;

			cudaFree(Dev_TestArrayIn);
			cudaFree(Dev_TestArrayOut);

			Host_thrust.clear();
			thrust::host_vector<double>().swap(Host_thrust);

			Device_thrust.clear();
			thrust::device_vector<double>().swap(Device_thrust);
		}

	}

	std::cout << "The total elapse time for bitonic sort is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerBitinicSort << std::endl;

	std::cout << "The total elapse time for arbitray bitonic sort is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalArbitraryBitonicSort << std::endl;

	std::cout << "The total elapse time for radix sort is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerRadixSort << std::endl;

	std::cout << "The total elapse time for thrust sort is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerThrust << std::endl;

	std::cout << "The total elapse time for CPU sort is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerCPU << std::endl;

	err = cudaSetDevice(currentDevice);

	return 0;
}


int main3(int argc, char** argv) {
	int NSize;
	double **Host_TestArrayIn;
	double *Host_TestArrayIn_X;
	double **Dev_TestArrayIn;
	double *Dev_TestArrayIn_SotedX_Shared;
	double *Dev_TestArrayIn_SotedX_noShared;
	double *TestArray_Dev_OneDim;
	double **Addr_HostRecordDev;
	int *SortedIndex_Dev_Shared;
	int *SortedIndex_Dev_noShared;
	int *SortedIndex_Host;
	int *Dev_NNearestNeighbor;
	int *Host_TestArrayOut_MyMehtod_RadixSort_Shared;
	int *Host_TestArrayOut_MyMehtod_RadixSort_noShared;
	int *Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort;
	int *Host_TestArrayOut_Norm;
	int *Host_TestArrayOut_CPU;
	int currentDevice;
	int noone;
	float totalTimerMyMethod_Shared;
	float timerMyMethod_Shared;
	float totalTimerMyMethod_noShared;
	float timerMyMethod_noShared;
	float totalTimerCommonGPU;
	float timerCommonGPU;

	totalTimerMyMethod_Shared = 0.E0;
	timerMyMethod_Shared = 0.E0;
	totalTimerMyMethod_noShared = 0.E0;
	timerMyMethod_noShared = 0.E0;
	totalTimerCommonGPU = 0.E0;
	timerCommonGPU = 0.E0;


	double BoxSize = 1000;

	cudaError err = cudaGetDevice(&currentDevice);

	err = cudaSetDevice(0);

	srand(55352);

	for (int NRand = 0; NRand < 1; NRand++) {
		for (NSize = 50; NSize < 100000000; NSize <<= 1) {

			std::cout << "NSize :  " << NSize << std::endl;

			Host_TestArrayIn = new double*[NSize];
			Addr_HostRecordDev = new double*[NSize];

			Host_TestArrayOut_MyMehtod_RadixSort_Shared = new int[NSize];
			Host_TestArrayOut_MyMehtod_RadixSort_noShared = new int[NSize];
			Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort = new int[NSize];
			Host_TestArrayOut_Norm = new int[NSize];
			Host_TestArrayOut_CPU = new int[NSize];

			Host_TestArrayIn_X = new double[NSize];

			SortedIndex_Host = new int[NSize];

			for (int i = 0; i < NSize; i++) {
				Host_TestArrayIn[i] = new double[3];

				for(int j=0;j<3;j++){
					Host_TestArrayIn[i][j] = BoxSize*(double(rand()) / RAND_MAX - 0.5);
				}

				Host_TestArrayIn_X[i] = Host_TestArrayIn[i][0];

				SortedIndex_Host[i] = i;
			}

			err = cudaMalloc((void**)&Dev_TestArrayIn, NSize * sizeof(double*));
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedX_Shared, NSize * sizeof(double));
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedX_noShared, NSize * sizeof(double));


			err = cudaMalloc((void**)&SortedIndex_Dev_Shared, NSize * sizeof(int));
			err = cudaMalloc((void**)&SortedIndex_Dev_noShared, NSize * sizeof(int));
			err = cudaMalloc((void**)&Dev_NNearestNeighbor, NSize * sizeof(int));

			for (int i = 0; i < NSize; i++) {
				err = cudaMalloc((void**)&TestArray_Dev_OneDim, 3 * sizeof(double));

				cudaMemcpy(TestArray_Dev_OneDim, Host_TestArrayIn[i], 3 * sizeof(double), cudaMemcpyHostToDevice);

				Addr_HostRecordDev[i] = TestArray_Dev_OneDim;
			}

			cudaMemcpy(Dev_TestArrayIn, Addr_HostRecordDev, NSize * sizeof(double*), cudaMemcpyHostToDevice);


			err = cudaMemcpy(Dev_TestArrayIn_SotedX_Shared,Host_TestArrayIn_X,NSize*sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(Dev_TestArrayIn_SotedX_noShared, Host_TestArrayIn_X, NSize * sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndex_Dev_Shared, SortedIndex_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndex_Dev_noShared, SortedIndex_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);

			if (cudaSuccess != err) {
				std::cout << "Error occour when copy  Dev_TestArrayIn_SotedX" << std::endl;

				std::cout << cudaGetErrorName(err) << std::endl;
				std::cout << cudaGetErrorString(err) << std::endl;
				std::cin >> noone;
			}


			std::cout << "************My Method radixSort shared************" << std::endl;

			My_NeighborListCal_RadixSort_Shared(NSize, Dev_TestArrayIn_SotedX_Shared, Dev_TestArrayIn, SortedIndex_Dev_Shared, Dev_NNearestNeighbor, Host_TestArrayOut_MyMehtod_RadixSort_Shared, timerMyMethod_Shared);

			totalTimerMyMethod_Shared = totalTimerMyMethod_Shared + timerMyMethod_Shared;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerMyMethod_Shared << " for My Method shared: " << NSize << std::endl;

			std::cout << "************My Method radixSort no shared************" << std::endl;

			My_NeighborListCal_RadixSort_noShared(NSize, Dev_TestArrayIn_SotedX_noShared, Dev_TestArrayIn, SortedIndex_Dev_noShared, Dev_NNearestNeighbor, Host_TestArrayOut_MyMehtod_RadixSort_noShared, timerMyMethod_noShared);

			totalTimerMyMethod_noShared = totalTimerMyMethod_noShared + timerMyMethod_noShared;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerMyMethod_noShared << " for My Method no shared: " << NSize << std::endl;

			std::cout << "************Method Common GPU************" << std::endl;

			Common_NeighborListCal(NSize, Dev_TestArrayIn,Dev_NNearestNeighbor, Host_TestArrayOut_Norm, timerCommonGPU);

			totalTimerCommonGPU = totalTimerCommonGPU + timerCommonGPU;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerCommonGPU << " for Common GPU: " << NSize << std::endl;

			//std::cout << "************Method Common CPU************" << std::endl;

			//Common_NeighborListCal_CPU(NSize, Host_TestArrayIn, Host_TestArrayOut_CPU);


			//Verify
			//for (int i = 0; i < NSize; i++) {
			//	if (Host_TestArrayOut_CPU[i] != Host_TestArrayOut_MyMehtod[i]) {
			//		std::cout << "It is wrong for index: " << i << std::endl;
			//		std::cout << "The CPU neighbor-list result is: " << Host_TestArrayOut_CPU[i] << std::endl;
			//		std::cout << "My neighbor-list calculation result is: " << Host_TestArrayOut_MyMehtod[i] << std::endl;

			//		std::cout << "  Distance normal GPU: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0]) +
			//			(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1]) +
			//			(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2]) << std::endl;

			//		std::cout << "My neighbor-list calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod[i]][0]) +
			//			(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod[i]][1]) +
			//			(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod[i]][2]) << std::endl;

			//		std::cin >> noone;
			//	}
			//}


			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_Norm[i] != Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list result is: " << Host_TestArrayOut_Norm[i] << std::endl;
					std::cout << "My neighbor-list calculation shared result is: " << Host_TestArrayOut_MyMehtod_RadixSort_Shared[i] << std::endl;

					std::cout <<"  Distance normal GPU: "<< (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2] << std::endl;

					std::cout << "My neighbor-list shared calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_Shared[i]][2] << std::endl;


					std::cin >> noone;
				}
			}


			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_Norm[i] != Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list result is: " << Host_TestArrayOut_Norm[i] << std::endl;
					std::cout << "My neighbor-list calculation  no shared result is: " << Host_TestArrayOut_MyMehtod_RadixSort_noShared[i] << std::endl;

					std::cout << "  Distance normal GPU: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2] << std::endl;

					std::cout << "My neighbor-list no shared calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort_noShared[i]][2] << std::endl;


					std::cin >> noone;
				}
			}

		}

	}


	std::cout << "The total elapse time for my method shared is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerMyMethod_Shared << std::endl;

	std::cout << "The total elapse time for my method no shared is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerMyMethod_noShared << std::endl;

	std::cout << "The total elapse time for Common GPU (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerCommonGPU << std::endl;


	err = cudaSetDevice(currentDevice);

	return 0;
}


int main(int argc, char** argv) {
	int NSize;
	double **Host_TestArrayIn;
	double *Host_TestArrayIn_X;
	double *Host_TestArrayIn_Y;
	double **Dev_TestArrayIn;
	double *Dev_TestArrayIn_SotedX_Shared_ForX;
	double *Dev_TestArrayIn_SotedX_noShared_ForX;
	double *Dev_TestArrayIn_SotedX_Shared_ForXY;
	double *Dev_TestArrayIn_SotedX_noShared_ForXY;
	double *Dev_TestArrayIn_SotedY_Shared_ForXY;
	double *Dev_TestArrayIn_SotedY_noShared_ForXY;
	double *TestArray_Dev_OneDim;
	double **Addr_HostRecordDev;
	int *SortedIndexX_Dev_Shared_ForX;
	int *SortedIndexX_Dev_noShared_ForX;
	int *SortedIndexX_Dev_Shared_ForXY;
	int *SortedIndexX_Dev_noShared_ForXY;
	int *SortedIndexY_Dev_Shared_ForXY;
	int *SortedIndexY_Dev_noShared_ForXY;
	int *ReverseMap_SortedIndexX_Dev_Shared_ForX;
	int *ReverseMap_SortedIndexX_Dev_noShared_ForX;
	int *ReverseMap_SortedIndexX_Dev_Shared_ForXY;
	int *ReverseMap_SortedIndexX_Dev_noShared_ForXY;
	int *ReverseMap_SortedIndexY_Dev_Shared_ForXY;
	int *ReverseMap_SortedIndexY_Dev_noShared_ForXY;
	int *SortedIndexX_Host;
	int *SortedIndexY_Host;
	int *ReverseMap_SortedIndexX_Host;
	int *ReverseMap_SortedIndexY_Host;
	int *Dev_NNearestNeighbor;
	int *Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared;
	int *Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared;
	int *Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared;
	int *Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared;
	int *Host_TestArrayOut_Norm_Shared;
	int *Host_TestArrayOut_Norm_noShared;
	int *Host_TestArrayOut_CPU;
	int **Host_IDStartEnd;
	int **Dev_IDStartEnd;
	int *TestArray_Dev_OneDim_StartEndID;
	int **Addr_HostRecordDev_StartEndID;
	int currentDevice;
	int noone;
	float totalTimerMyMethod_SortX_Shared;
	float timerMyMethod_SortX_Shared;
	float totalTimerMyMethod_SortX_noShared;
	float timerMyMethod_SortX_noShared;
	float totalTimerMyMethod_SortXY_Shared;
	float timerMyMethod_SortXY_Shared;
	float totalTimerMyMethod_SortXY_noShared;
	float timerMyMethod_SortXY_noShared;
	float totalTimerCommonGPU_Shared;
	float timerCommonGPU_Shared;
	float totalTimerCommonGPU_noShared;
	float timerCommonGPU_noShared;

	totalTimerMyMethod_SortX_Shared = 0.E0;
	timerMyMethod_SortX_Shared = 0.E0;
	totalTimerMyMethod_SortX_noShared = 0.E0;
	timerMyMethod_SortX_noShared = 0.E0;
	totalTimerCommonGPU_Shared = 0.E0;
	timerCommonGPU_Shared = 0.E0;
	totalTimerCommonGPU_noShared = 0.E0;
	timerCommonGPU_noShared = 0.E0;


	double BoxSize = 1000;
	int NBox = 800;

	cudaError err = cudaGetDevice(&currentDevice);

	err = cudaSetDevice(0);

	srand(55352);

	for (int NRand = 0; NRand < 1; NRand++) {

		for (int NSizeEachBox = 50; NSizeEachBox < 100000000; NSizeEachBox <<= 1) {

			NSize = NSizeEachBox * NBox;

			std::cout << "NSize :  " << NSize << std::endl;

			Host_TestArrayIn = new double*[NSize];
			Addr_HostRecordDev = new double*[NSize];

			Host_IDStartEnd = new int*[NBox];
			Addr_HostRecordDev_StartEndID = new int*[NBox];

			Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared = new int[NSize];
			Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared = new int[NSize];
			Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared = new int[NSize];
			Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared = new int[NSize];
			Host_TestArrayOut_Norm_Shared = new int[NSize];
			Host_TestArrayOut_Norm_noShared = new int[NSize];
			Host_TestArrayOut_CPU = new int[NSize];

			Host_TestArrayIn_X = new double[NSize];
			Host_TestArrayIn_Y = new double[NSize];

			SortedIndexX_Host = new int[NSize];
			SortedIndexY_Host = new int[NSize];

			ReverseMap_SortedIndexX_Host = new int[NSize];
			ReverseMap_SortedIndexY_Host = new int[NSize];

			for (int i = 0; i < NSize; i++) {
				Host_TestArrayIn[i] = new double[3];

				for (int j = 0; j < 3; j++) {
					Host_TestArrayIn[i][j] = BoxSize * (double(rand()) / RAND_MAX - 0.5);
				}

				Host_TestArrayIn_X[i] = Host_TestArrayIn[i][0];
				Host_TestArrayIn_Y[i] = Host_TestArrayIn[i][1];

				SortedIndexX_Host[i] = i;
				SortedIndexY_Host[i] = i;

				ReverseMap_SortedIndexX_Host[i] = i;
				ReverseMap_SortedIndexY_Host[i] = i;
			}


			err = cudaMalloc((void**)&Dev_IDStartEnd,NBox*sizeof(int*));
			for (int i = 0; i < NBox; i++) {
				Host_IDStartEnd[i] = new int[2];

				Host_IDStartEnd[i][0] = i * NSizeEachBox;

				Host_IDStartEnd[i][1] = (i+1)*NSizeEachBox - 1;

				if (Host_IDStartEnd[i][1] < 0) Host_IDStartEnd[i][1] = 0;

				err = cudaMalloc((void**)&TestArray_Dev_OneDim_StartEndID, 2 * sizeof(int));
				
				cudaMemcpy(TestArray_Dev_OneDim_StartEndID, Host_IDStartEnd[i], 2 * sizeof(int), cudaMemcpyHostToDevice);

				Addr_HostRecordDev_StartEndID[i] = TestArray_Dev_OneDim_StartEndID;
			}

			cudaMemcpy(Dev_IDStartEnd, Addr_HostRecordDev_StartEndID, NBox *sizeof(int*), cudaMemcpyHostToDevice);


			err = cudaMalloc((void**)&Dev_TestArrayIn, NSize * sizeof(double*));
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedX_Shared_ForX, NSize * sizeof(double));
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedX_noShared_ForX, NSize * sizeof(double));
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedX_Shared_ForXY, NSize * sizeof(double));
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedX_noShared_ForXY, NSize * sizeof(double));
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedY_Shared_ForXY, NSize * sizeof(double));
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedY_noShared_ForXY, NSize * sizeof(double));


			err = cudaMalloc((void**)&SortedIndexX_Dev_Shared_ForX, NSize * sizeof(int));
			err = cudaMalloc((void**)&SortedIndexX_Dev_noShared_ForX, NSize * sizeof(int));
			err = cudaMalloc((void**)&SortedIndexX_Dev_Shared_ForXY, NSize * sizeof(int));
			err = cudaMalloc((void**)&SortedIndexX_Dev_noShared_ForXY, NSize * sizeof(int));
			err = cudaMalloc((void**)&SortedIndexY_Dev_Shared_ForXY, NSize * sizeof(int));
			err = cudaMalloc((void**)&SortedIndexY_Dev_noShared_ForXY, NSize * sizeof(int));

			err = cudaMalloc((void**)&ReverseMap_SortedIndexX_Dev_Shared_ForX, NSize * sizeof(int));
			err = cudaMalloc((void**)&ReverseMap_SortedIndexX_Dev_noShared_ForX, NSize * sizeof(int));
			err = cudaMalloc((void**)&ReverseMap_SortedIndexX_Dev_Shared_ForXY, NSize * sizeof(int));
			err = cudaMalloc((void**)&ReverseMap_SortedIndexX_Dev_noShared_ForXY, NSize * sizeof(int));
			err = cudaMalloc((void**)&ReverseMap_SortedIndexY_Dev_Shared_ForXY, NSize * sizeof(int));
			err = cudaMalloc((void**)&ReverseMap_SortedIndexY_Dev_noShared_ForXY, NSize * sizeof(int));


			err = cudaMalloc((void**)&Dev_NNearestNeighbor, NSize * sizeof(int));

			for (int i = 0; i < NSize; i++) {
				err = cudaMalloc((void**)&TestArray_Dev_OneDim, 3 * sizeof(double));

				cudaMemcpy(TestArray_Dev_OneDim, Host_TestArrayIn[i], 3 * sizeof(double), cudaMemcpyHostToDevice);

				Addr_HostRecordDev[i] = TestArray_Dev_OneDim;
			}

			cudaMemcpy(Dev_TestArrayIn, Addr_HostRecordDev, NSize * sizeof(double*), cudaMemcpyHostToDevice);


			err = cudaMemcpy(Dev_TestArrayIn_SotedX_Shared_ForX, Host_TestArrayIn_X, NSize * sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(Dev_TestArrayIn_SotedX_noShared_ForX, Host_TestArrayIn_X, NSize * sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(Dev_TestArrayIn_SotedX_Shared_ForXY, Host_TestArrayIn_X, NSize * sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(Dev_TestArrayIn_SotedX_noShared_ForXY, Host_TestArrayIn_X, NSize * sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(Dev_TestArrayIn_SotedY_Shared_ForXY, Host_TestArrayIn_Y, NSize * sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(Dev_TestArrayIn_SotedY_noShared_ForXY, Host_TestArrayIn_Y, NSize * sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndexX_Dev_Shared_ForX, SortedIndexX_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndexX_Dev_noShared_ForX, SortedIndexX_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndexX_Dev_Shared_ForXY, SortedIndexX_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndexX_Dev_noShared_ForXY, SortedIndexX_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndexY_Dev_Shared_ForXY, SortedIndexY_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndexY_Dev_noShared_ForXY, SortedIndexY_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(ReverseMap_SortedIndexX_Dev_Shared_ForX, ReverseMap_SortedIndexX_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(ReverseMap_SortedIndexX_Dev_noShared_ForX, ReverseMap_SortedIndexX_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(ReverseMap_SortedIndexX_Dev_Shared_ForXY, ReverseMap_SortedIndexX_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(ReverseMap_SortedIndexX_Dev_noShared_ForXY, ReverseMap_SortedIndexX_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(ReverseMap_SortedIndexY_Dev_Shared_ForXY, ReverseMap_SortedIndexY_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);
			err = cudaMemcpy(ReverseMap_SortedIndexY_Dev_noShared_ForXY, ReverseMap_SortedIndexY_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);

			if (cudaSuccess != err) {
				std::cout << "Error occour when copy  Dev_TestArrayIn_SotedX" << std::endl;

				std::cout << cudaGetErrorName(err) << std::endl;
				std::cout << cudaGetErrorString(err) << std::endl;
				std::cin >> noone;
			}


			std::cout << "************My Method Arbitaru bitinic sort X Shared************" << std::endl;

			My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_Shared(NSize,NBox, Host_IDStartEnd, Dev_IDStartEnd, Dev_TestArrayIn_SotedX_Shared_ForX, Dev_TestArrayIn, SortedIndexX_Dev_Shared_ForX, ReverseMap_SortedIndexX_Dev_Shared_ForX, Dev_NNearestNeighbor, Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared, timerMyMethod_SortX_Shared);

			totalTimerMyMethod_SortX_Shared = totalTimerMyMethod_SortX_Shared + timerMyMethod_SortX_Shared;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerMyMethod_SortX_Shared << " for My Method sort X shared: " << NSize << std::endl;


			std::cout << "************My Method Arbitaru bitinic sort X no Shared************" << std::endl;

			My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_noShared(NSize, NBox, Host_IDStartEnd, Dev_IDStartEnd, Dev_TestArrayIn_SotedX_noShared_ForX, Dev_TestArrayIn, SortedIndexX_Dev_noShared_ForX, ReverseMap_SortedIndexX_Dev_Shared_ForX, Dev_NNearestNeighbor, Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared, timerMyMethod_SortX_noShared);

			totalTimerMyMethod_SortX_noShared = totalTimerMyMethod_SortX_noShared + timerMyMethod_SortX_noShared;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerMyMethod_SortX_noShared << " for My Method sort X no shared: " << NSize << std::endl;


			std::cout << "************My Method Arbitaru bitinic sort XY Shared************" << std::endl;

			My_NeighborListCal_ArbitrayBitonicSortXY_multipleBox_Shared(NSize,
																	    NBox, 
																		Host_IDStartEnd, 
																		Dev_IDStartEnd, 
																		Dev_TestArrayIn_SotedX_Shared_ForXY, 
																		Dev_TestArrayIn_SotedY_Shared_ForXY,
																		Dev_TestArrayIn, 
																		SortedIndexX_Dev_Shared_ForXY,
																		SortedIndexY_Dev_Shared_ForXY,
																		ReverseMap_SortedIndexX_Dev_Shared_ForXY,
																		ReverseMap_SortedIndexY_Dev_Shared_ForXY,
																		Dev_NNearestNeighbor, 
																		Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared, 
																		timerMyMethod_SortXY_Shared);

			totalTimerMyMethod_SortXY_Shared = totalTimerMyMethod_SortXY_Shared + timerMyMethod_SortXY_Shared;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerMyMethod_SortXY_Shared << " for My Method sort XY shared: " << NSize << std::endl;

			std::cout << "************My Method Arbitaru bitinic sort XY no Shared************" << std::endl;

			My_NeighborListCal_ArbitrayBitonicSortXY_multipleBox_noShared(NSize, 
																		  NBox, 
																		  Host_IDStartEnd, 
																		  Dev_IDStartEnd, 
																		  Dev_TestArrayIn_SotedX_noShared_ForXY,
																		  Dev_TestArrayIn_SotedY_noShared_ForXY,
																		  Dev_TestArrayIn, 
																		  SortedIndexX_Dev_noShared_ForXY,
																		  SortedIndexY_Dev_noShared_ForXY,
																		  ReverseMap_SortedIndexX_Dev_noShared_ForXY,
																		  ReverseMap_SortedIndexY_Dev_noShared_ForXY,
																		  Dev_NNearestNeighbor, 
																		  Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared, 
																		  timerMyMethod_SortXY_noShared);

			totalTimerMyMethod_SortXY_noShared = totalTimerMyMethod_SortXY_noShared + timerMyMethod_SortXY_noShared;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerMyMethod_SortXY_noShared << " for My Method sort XY no shared: " << NSize << std::endl;


			std::cout << "************Method Common GPU shared************" << std::endl;

			Common_NeighborListCal_multipleBox_Shared(NSize, NBox, Host_IDStartEnd, Dev_IDStartEnd, Dev_TestArrayIn, Dev_NNearestNeighbor, Host_TestArrayOut_Norm_Shared, timerCommonGPU_Shared);

			totalTimerCommonGPU_Shared = totalTimerCommonGPU_Shared + timerCommonGPU_Shared;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerCommonGPU_Shared << " for Common GPU shared: " << NSize << std::endl;

			std::cout << "************Method Common GPU no shared************" << std::endl;

			Common_NeighborListCal_multipleBox_noShared(NSize, NBox, Host_IDStartEnd, Dev_IDStartEnd, Dev_TestArrayIn, Dev_NNearestNeighbor, Host_TestArrayOut_Norm_noShared, timerCommonGPU_noShared);

			totalTimerCommonGPU_noShared = totalTimerCommonGPU_noShared + timerCommonGPU_noShared;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerCommonGPU_noShared << " for Common GPU no shared: " << NSize << std::endl;

			//std::cout << "************Method Common CPU************" << std::endl;

			//Common_NeighborListCal_CPU_multipleBox(NSize, NBox, Host_IDStartEnd, Host_TestArrayIn, Host_TestArrayOut_CPU);

			/*
			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_CPU[i] != Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_Shared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The CPU neighbor-list result is: " << Host_TestArrayOut_CPU[i] << std::endl;
					std::cout << "My neighbor-list shared calculation result is: " << Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_Shared[i] << std::endl;

					std::cout << "  Distance CPU: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2]) << std::endl;

					std::cout << "My neighbor-list shared calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_Shared[i]][2]) << std::endl;

					std::cin >> noone;
				}
			}

			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_CPU[i] != Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_noShared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The CPU neighbor-list result is: " << Host_TestArrayOut_CPU[i] << std::endl;
					std::cout << "My neighbor-list no shared calculation result is: " << Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_noShared[i] << std::endl;

					std::cout << "  Distance CPU: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2]) << std::endl;

					std::cout << "My neighbor-list no shared calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_noShared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_noShared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_noShared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_noShared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_noShared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort_noShared[i]][2]) << std::endl;

					std::cin >> noone;
				}
			}
			

			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_CPU[i] != Host_TestArrayOut_Norm[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The CPU neighbor-list result is: " << Host_TestArrayOut_CPU[i] << std::endl;
					std::cout << "The common GPU result is: " << Host_TestArrayOut_Norm[i] << std::endl;

					std::cout << "   Distance CPU: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2]) << std::endl;

					std::cout << "The common GPU calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2]) << std::endl;

					std::cin >> noone;
				}
			}

			*/
			

			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_Norm_Shared[i] != Host_TestArrayOut_Norm_noShared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list shared result is: " << Host_TestArrayOut_Norm_Shared[i] << std::endl;
					std::cout << "The common GPU neighbor-list no shared result is: " << Host_TestArrayOut_Norm_noShared[i] << std::endl;

					std::cout << "  Distance the common GPU neighbor-list shared: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2] << std::endl;

					std::cout << " Distance the common GPU neighbor-list no shared: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_noShared[i]][2] << std::endl;


					std::cin >> noone;
				}
			}


			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_Norm_Shared[i] != Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list shared result is: " << Host_TestArrayOut_Norm_Shared[i] << std::endl;
					std::cout << "My neighbor-list sort X shared calculation result is: " << Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i] << std::endl;

					std::cout << "  Distance the common GPU neighbor-list shared: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2] << std::endl;

					std::cout << "My neighbor-list sort X shared  calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_Shared[i]][2] << std::endl;


					std::cin >> noone;
				}
			}

			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_Norm_Shared[i] != Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list shared result is: " << Host_TestArrayOut_Norm_Shared[i] << std::endl;
					std::cout << "My neighbor-list sort X no shared calculation result is: " << Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i] << std::endl;

					std::cout << "  Distance the common GPU neighbor-list shared : " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2] << std::endl;

					std::cout << "My neighbor-list sort X no shared calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortX_noShared[i]][2] << std::endl;


					std::cin >> noone;
				}
			}


			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_Norm_Shared[i] != Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list shared result is: " << Host_TestArrayOut_Norm_Shared[i] << std::endl;
					std::cout << "My neighbor-list sort XY shared calculation result is: " << Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i] << std::endl;

					std::cout << "  Distance the common GPU neighbor-list shared: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2] << std::endl;

					std::cout << "My neighbor-list sort XY shared  calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_Shared[i]][2] << std::endl;


					std::cin >> noone;
				}
			}

			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_Norm_Shared[i] != Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list shared result is: " << Host_TestArrayOut_Norm_Shared[i] << std::endl;
					std::cout << "My neighbor-list sort XY no shared calculation result is: " << Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i] << std::endl;

					std::cout << "  Distance the common GPU neighbor-list shared : " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm_Shared[i]][2] << std::endl;

					std::cout << "My neighbor-list sort XY no shared calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSortXY_noShared[i]][2] << std::endl;


					std::cin >> noone;
				}
			}
			

		}

	}


	std::cout << "The total elapse time for my method sort X shared is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerMyMethod_SortX_Shared << std::endl;

	std::cout << "The total elapse time for my method sort X no shared is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerMyMethod_SortX_noShared << std::endl;

	std::cout << "The total elapse time for my method sort XY shared is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerMyMethod_SortXY_Shared << std::endl;

	std::cout << "The total elapse time for my method sort XY no shared is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerMyMethod_SortXY_noShared << std::endl;

	std::cout << "The total elapse time for Common GPU shared (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerCommonGPU_Shared << std::endl;


	err = cudaSetDevice(currentDevice);

	return 0;
}