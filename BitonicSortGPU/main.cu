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
	double *Dev_TestArrayIn_SotedX;
	double *TestArray_Dev_OneDim;
	double **Addr_HostRecordDev;
	int *SortedIndex_Dev;
	int *SortedIndex_Host;
	int *Dev_NNearestNeighbor;
	int *Host_TestArrayOut_MyMehtod_RadixSort;
	int *Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort;
	int *Host_TestArrayOut_Norm;
	int *Host_TestArrayOut_CPU;
	int currentDevice;
	int noone;
	float totalTimerMyMethod;
	float timerMyMethod;
	float totalTimerCommonGPU;
	float timerCommonGPU;

	totalTimerMyMethod = 0.E0;
	timerMyMethod = 0.E0;
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

			Host_TestArrayOut_MyMehtod_RadixSort = new int[NSize];
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
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedX, NSize * sizeof(double));


			err = cudaMalloc((void**)&SortedIndex_Dev, NSize * sizeof(int));
			err = cudaMalloc((void**)&Dev_NNearestNeighbor, NSize * sizeof(int));

			for (int i = 0; i < NSize; i++) {
				err = cudaMalloc((void**)&TestArray_Dev_OneDim, 3 * sizeof(double));

				cudaMemcpy(TestArray_Dev_OneDim, Host_TestArrayIn[i], 3 * sizeof(double), cudaMemcpyHostToDevice);

				Addr_HostRecordDev[i] = TestArray_Dev_OneDim;
			}

			cudaMemcpy(Dev_TestArrayIn, Addr_HostRecordDev, NSize * sizeof(double*), cudaMemcpyHostToDevice);


			err = cudaMemcpy(Dev_TestArrayIn_SotedX,Host_TestArrayIn_X,NSize*sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndex_Dev, SortedIndex_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);


			if (cudaSuccess != err) {
				std::cout << "Error occour when copy  Dev_TestArrayIn_SotedX" << std::endl;

				std::cout << cudaGetErrorName(err) << std::endl;
				std::cout << cudaGetErrorString(err) << std::endl;
				std::cin >> noone;
			}


			std::cout << "************My Method radixSort************" << std::endl;

			My_NeighborListCal_RadixSort(NSize, Dev_TestArrayIn_SotedX, Dev_TestArrayIn, SortedIndex_Dev, Dev_NNearestNeighbor, Host_TestArrayOut_MyMehtod_RadixSort, timerMyMethod);

			totalTimerMyMethod = totalTimerMyMethod + timerMyMethod;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerMyMethod << " for My Method: " << NSize << std::endl;

			std::cout << "************Method Common GPU************" << std::endl;

			Common_NeighborListCal(NSize, Dev_TestArrayIn,Dev_NNearestNeighbor, Host_TestArrayOut_Norm, timerCommonGPU);

			totalTimerCommonGPU = totalTimerCommonGPU + timerCommonGPU;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerCommonGPU << " for Common GPU: " << NSize << std::endl;

			std::cout << "************Method Common CPU************" << std::endl;

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
				if (Host_TestArrayOut_Norm[i] != Host_TestArrayOut_MyMehtod_RadixSort[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list result is: " << Host_TestArrayOut_Norm[i] << std::endl;
					std::cout << "My neighbor-list calculation result is: " << Host_TestArrayOut_MyMehtod_RadixSort[i] << std::endl;

					std::cout <<"  Distance normal GPU: "<< (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2] << std::endl;

					std::cout << "My neighbor-list calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][2] << std::endl;


					std::cin >> noone;
				}
			}

		}

	}


	std::cout << "The total elapse time for my method is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerMyMethod << std::endl;

	std::cout << "The total elapse time for Common GPU (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerCommonGPU << std::endl;


	err = cudaSetDevice(currentDevice);

	return 0;
}


int main(int argc, char** argv) {
	int NSize;
	double **Host_TestArrayIn;
	double *Host_TestArrayIn_X;
	double **Dev_TestArrayIn;
	double *Dev_TestArrayIn_SotedX;
	double *TestArray_Dev_OneDim;
	double **Addr_HostRecordDev;
	int *SortedIndex_Dev;
	int *SortedIndex_Host;
	int *Dev_NNearestNeighbor;
	int *Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort;
	int *Host_TestArrayOut_Norm;
	int *Host_TestArrayOut_CPU;
	int **Host_IDStartEnd;
	int **Dev_IDStartEnd;
	int *TestArray_Dev_OneDim_StartEndID;
	int **Addr_HostRecordDev_StartEndID;
	int currentDevice;
	int noone;
	float totalTimerMyMethod;
	float timerMyMethod;
	float totalTimerCommonGPU;
	float timerCommonGPU;

	totalTimerMyMethod = 0.E0;
	timerMyMethod = 0.E0;
	totalTimerCommonGPU = 0.E0;
	timerCommonGPU = 0.E0;


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

			Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort = new int[NSize];
			Host_TestArrayOut_Norm = new int[NSize];
			Host_TestArrayOut_CPU = new int[NSize];

			Host_TestArrayIn_X = new double[NSize];

			SortedIndex_Host = new int[NSize];

			for (int i = 0; i < NSize; i++) {
				Host_TestArrayIn[i] = new double[3];

				for (int j = 0; j < 3; j++) {
					Host_TestArrayIn[i][j] = BoxSize * (double(rand()) / RAND_MAX - 0.5);
				}

				Host_TestArrayIn_X[i] = Host_TestArrayIn[i][0];

				SortedIndex_Host[i] = i;
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
			err = cudaMalloc((void**)&Dev_TestArrayIn_SotedX, NSize * sizeof(double));


			err = cudaMalloc((void**)&SortedIndex_Dev, NSize * sizeof(int));
			err = cudaMalloc((void**)&Dev_NNearestNeighbor, NSize * sizeof(int));

			for (int i = 0; i < NSize; i++) {
				err = cudaMalloc((void**)&TestArray_Dev_OneDim, 3 * sizeof(double));

				cudaMemcpy(TestArray_Dev_OneDim, Host_TestArrayIn[i], 3 * sizeof(double), cudaMemcpyHostToDevice);

				Addr_HostRecordDev[i] = TestArray_Dev_OneDim;
			}

			cudaMemcpy(Dev_TestArrayIn, Addr_HostRecordDev, NSize * sizeof(double*), cudaMemcpyHostToDevice);


			err = cudaMemcpy(Dev_TestArrayIn_SotedX, Host_TestArrayIn_X, NSize * sizeof(double), cudaMemcpyHostToDevice);
			err = cudaMemcpy(SortedIndex_Dev, SortedIndex_Host, NSize * sizeof(int), cudaMemcpyHostToDevice);


			if (cudaSuccess != err) {
				std::cout << "Error occour when copy  Dev_TestArrayIn_SotedX" << std::endl;

				std::cout << cudaGetErrorName(err) << std::endl;
				std::cout << cudaGetErrorString(err) << std::endl;
				std::cin >> noone;
			}


			std::cout << "************My Method Arbitaru bitinic sort************" << std::endl;

			My_NeighborListCal_ArbitrayBitonicSort_multipleBox(NSize,NBox, Host_IDStartEnd, Dev_IDStartEnd, Dev_TestArrayIn_SotedX, Dev_TestArrayIn, SortedIndex_Dev, Dev_NNearestNeighbor, Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort, timerMyMethod);

			totalTimerMyMethod = totalTimerMyMethod + timerMyMethod;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerMyMethod << " for My Method: " << NSize << std::endl;

			std::cout << "************Method Common GPU************" << std::endl;

			Common_NeighborListCal_multipleBox(NSize, NBox, Host_IDStartEnd, Dev_IDStartEnd, Dev_TestArrayIn, Dev_NNearestNeighbor, Host_TestArrayOut_Norm, timerCommonGPU);

			totalTimerCommonGPU = totalTimerCommonGPU + timerCommonGPU;

			std::cout << "The elapse time is (ms): " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << timerCommonGPU << " for Common GPU: " << NSize << std::endl;

			std::cout << "************Method Common CPU************" << std::endl;

			Common_NeighborListCal_CPU_multipleBox(NSize, NBox, Host_IDStartEnd, Host_TestArrayIn, Host_TestArrayOut_CPU);

			
			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_CPU[i] != Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The CPU neighbor-list result is: " << Host_TestArrayOut_CPU[i] << std::endl;
					std::cout << "My neighbor-list calculation result is: " << Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort[i] << std::endl;

					std::cout << "  Distance CPU: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_CPU[i]][2]) << std::endl;

					std::cout << "My neighbor-list calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_ArbitrayBitonicSort[i]][2]) << std::endl;

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
			/*
			//Verify
			for (int i = 0; i < NSize; i++) {
				if (Host_TestArrayOut_Norm[i] != Host_TestArrayOut_MyMehtod_RadixSort[i]) {
					std::cout << "It is wrong for index: " << i << std::endl;
					std::cout << "The common GPU neighbor-list result is: " << Host_TestArrayOut_Norm[i] << std::endl;
					std::cout << "My neighbor-list calculation result is: " << Host_TestArrayOut_MyMehtod_RadixSort[i] << std::endl;

					std::cout << "  Distance normal GPU: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2]) << std::endl;


					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_Norm[i]][2] << std::endl;

					std::cout << "My neighbor-list calculation: " << (Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][0])*(Host_TestArrayIn[i][0] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][0]) +
						(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][1])*(Host_TestArrayIn[i][1] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][1]) +
						(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][2])*(Host_TestArrayIn[i][2] - Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][2]) << std::endl;

					std::cout << Host_TestArrayIn[i][0] << " " << Host_TestArrayIn[i][1] << " " << Host_TestArrayIn[i][2] << std::endl;
					std::cout << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][0] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][1] << " " << Host_TestArrayIn[Host_TestArrayOut_MyMehtod_RadixSort[i]][2] << std::endl;


					std::cin >> noone;
				}
			}
			*/

		}

	}


	std::cout << "The total elapse time for my method is (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerMyMethod << std::endl;

	std::cout << "The total elapse time for Common GPU (ms) : " << std::setiosflags(std::ios::fixed) << std::setprecision(8) << totalTimerCommonGPU << std::endl;


	err = cudaSetDevice(currentDevice);

	return 0;
}