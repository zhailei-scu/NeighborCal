#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_atomic_functions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include "MyNeighborList_multipleBox.h"
#include <algorithm>

#define BLOCKSIZE 512

__global__ void Kernel_MyNeighborListCal_multipleBox(int NClusters, int NBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int LeftBound;
	int RightBound;
	int sortedID;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int relativeIC;
	double distance;
	double minDistance;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	bool exitFlag;
	int NRemind;
	int MapedIdex;
	__shared__ double Shared_XYZ[BLOCKSIZE][3];
	__shared__ int Shared_SortedID[BLOCKSIZE];
	__shared__ int NExitedThreadsRight;
	__shared__ int NExitedThreadsLeft;

	minDistance = 1.E32;

	/*Right Hand Searching*/
	exitFlag = false;
	LeftBound = bid * BLOCKSIZE;
	if (LeftBound < 0) LeftBound = 0;
	RightBound = (bid + 1)*BLOCKSIZE;
	if (RightBound > NClusters) RightBound = NClusters;

	NRemind = RightBound - LeftBound;

	NExitedThreadsRight = 0;

	if (cid < NClusters) {

		MapedIdex = SortedIndex[cid];

		Pos_X = Dev_ClustersPosXYZ[MapedIdex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIdex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIdex][2];

	}

	while (LeftBound < RightBound) {

		if (NExitedThreadsRight >= NRemind) {
			break;
		}

		if ((LeftBound + tid) < NClusters) {

			sortedID = SortedIndex[LeftBound + tid];

			Shared_SortedID[tid] = sortedID;

			Shared_XYZ[tid][0] = Dev_ClustersPosXYZ[sortedID][0];

			Shared_XYZ[tid][1] = Dev_ClustersPosXYZ[sortedID][1];
			Shared_XYZ[tid][2] = Dev_ClustersPosXYZ[sortedID][2];
		}

		__syncthreads();


		if (cid < NClusters) {

			if (false == exitFlag) {
				for (int IC = LeftBound; IC < RightBound; IC++) {
					if (IC != cid) {

						relativeIC = IC - LeftBound;

						distanceX = Shared_XYZ[relativeIC][0] - Pos_X;
						distanceY = Shared_XYZ[relativeIC][1] - Pos_Y;
						distanceZ = Shared_XYZ[relativeIC][2] - Pos_Z;

						distanceX = distanceX * distanceX;
						distanceY = distanceY * distanceY;
						distanceZ = distanceZ * distanceZ;

						distance = distanceX + distanceY + distanceZ;

						if (minDistance > distance) {
							minDistance = distance;
							NNID = Shared_SortedID[relativeIC];
						}

						if (distanceX > minDistance) {
							exitFlag = true;
							atomicAdd_block(&NExitedThreadsRight, 1);
							break;
						}

					}

				}
			}

		}

		__syncthreads();

		LeftBound = LeftBound + BLOCKSIZE;
		RightBound = RightBound + BLOCKSIZE;
		if (RightBound > NClusters) RightBound = NClusters;
	}

	/*Left Hand Searching*/
	exitFlag = false;
	LeftBound = (bid - 1) * BLOCKSIZE;
	if (LeftBound < 0) LeftBound = 0;
	RightBound = bid * BLOCKSIZE;
	if (RightBound > NClusters) RightBound = NClusters;

	NExitedThreadsLeft = 0;

	while (LeftBound < RightBound) {

		if (NExitedThreadsLeft >= NRemind) {
			break;
		}


		if ((LeftBound + tid) < NClusters) {

			sortedID = SortedIndex[LeftBound + tid];

			Shared_SortedID[tid] = sortedID;

			Shared_XYZ[tid][0] = Dev_ClustersPosXYZ[sortedID][0];

			Shared_XYZ[tid][1] = Dev_ClustersPosXYZ[sortedID][1];
			Shared_XYZ[tid][2] = Dev_ClustersPosXYZ[sortedID][2];

		}

		__syncthreads();

		if (cid < NClusters) {

			if (false == exitFlag) {
				for (int IC = RightBound - 1; IC >= LeftBound; IC--) {
					if (IC != cid) {

						relativeIC = IC - LeftBound;

						distanceX = Shared_XYZ[relativeIC][0] - Pos_X;
						distanceY = Shared_XYZ[relativeIC][1] - Pos_Y;
						distanceZ = Shared_XYZ[relativeIC][2] - Pos_Z;

						distanceX = distanceX * distanceX;
						distanceY = distanceY * distanceY;
						distanceZ = distanceZ * distanceZ;

						distance = distanceX + distanceY + distanceZ;

						if (minDistance > distance) {
							minDistance = distance;
							NNID = Shared_SortedID[relativeIC];
						}

						if (distanceX > minDistance) {
							exitFlag = true;
							atomicAdd_block(&NExitedThreadsLeft, 1);
							break;
						}

					}

				}
			}

		}

		__syncthreads();

		RightBound = LeftBound;
		LeftBound = LeftBound - BLOCKSIZE;
		if (LeftBound < 0) LeftBound = 0;
	}


	if (cid < NClusters) {
		Dev_NNearestNeighbor[MapedIdex] = NNID;
	}

}



__global__ void Kernel_NormalCalcNeighborList_multipleBox(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int LeftBound;
	int RightBound;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int relativeJC;
	double distance;
	double minDistance;
	int NNID;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int IC;
	__shared__ double Shared_XYZ[BLOCKSIZE][3];

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox* BlockNumEachBox;

	IC = scid + (cid - bid0* BLOCKSIZE);

	minDistance = 1.E32;

	/*Right Hand Searching*/
	LeftBound = scid;
	RightBound = LeftBound + BLOCKSIZE -1;
	//RightBound = RightBound < NClusters ? RightBound : NClusters;

	if (RightBound > ecid) RightBound = ecid;

	if (IC <= ecid) {
		Pos_X = Dev_ClustersPosXYZ[IC][0];
		Pos_Y = Dev_ClustersPosXYZ[IC][1];
		Pos_Z = Dev_ClustersPosXYZ[IC][2];
	}

	while (LeftBound < RightBound) {

		if ((LeftBound + tid) <= ecid) {
			Shared_XYZ[tid][0] = Dev_ClustersPosXYZ[LeftBound + tid][0];
			Shared_XYZ[tid][1] = Dev_ClustersPosXYZ[LeftBound + tid][1];
			Shared_XYZ[tid][2] = Dev_ClustersPosXYZ[LeftBound + tid][2];
		}

		__syncthreads();

		if (IC <= ecid) {

			for (int JC = LeftBound; JC <= RightBound; JC++) {
				if (JC != IC) {

					relativeJC = JC - LeftBound;

					distance = (Shared_XYZ[relativeJC][0] - Pos_X)*(Shared_XYZ[relativeJC][0] - Pos_X) +
						(Shared_XYZ[relativeJC][1] - Pos_Y)*(Shared_XYZ[relativeJC][1] - Pos_Y) +
						(Shared_XYZ[relativeJC][2] - Pos_Z)*(Shared_XYZ[relativeJC][2] - Pos_Z);

					if (distance < minDistance) {
						NNID = JC;

						minDistance = distance;
					}

				}

			}

		}

		__syncthreads();

		LeftBound = RightBound + 1;
		RightBound = RightBound + BLOCKSIZE - 1;
		//RightBound = RightBound < NClusters ? RightBound : NClusters;
		if (RightBound > ecid) RightBound = ecid;

	}

	if (IC <= ecid) {
		Dev_NNearestNeighbor[IC] = NNID;
	}

}



void My_NeighborListCal_RadixSort_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	thrust::device_ptr<double> Device_thrust_Key(ToSortDev_ClustersPosX);
	thrust::device_ptr<int> Device_thrust_Value(SortedIndex);

	NB = (NClusters - 1) / BLOCKSIZE + 1;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);
	thrust::sort_by_key(Device_thrust_Key, Device_thrust_Key + NClusters, Device_thrust_Value);

	Kernel_MyNeighborListCal_multipleBox << < blocks, threads >> > (NClusters, NBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, SortedIndex, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);

}

//void My_NeighborListCal_ArbitrayBitonicSort(int NClusters, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
//	dim3 threads;
//	dim3 blocks;
//	int NB;
//	cudaError err;
//	int noone;
//
//	cudaEvent_t StartEvent;
//	cudaEvent_t StopEvent;
//
//	cudaEventCreate(&StartEvent);
//	cudaEventCreate(&StopEvent);
//
//	cudaEventRecord(StartEvent, 0);
//
//	thrust::device_ptr<double> Device_thrust_Key(ToSortDev_ClustersPosX);
//	thrust::device_ptr<int> Device_thrust_Value(SortedIndex);
//
//	NB = (NClusters - 1) / BLOCKSIZE + 1;
//
//	blocks = dim3(NB, 1, 1);
//	threads = dim3(BLOCKSIZE, 1, 1);
//	thrust::sort_by_key(Device_thrust_Key, Device_thrust_Key + NClusters, Device_thrust_Value);
//
//	Kernel_MyNeighborListCal << < blocks, threads >> > (NClusters, Dev_ClustersPosXYZ, SortedIndex, Dev_NNearestNeighbor);
//
//	cudaDeviceSynchronize();
//
//	cudaEventRecord(StopEvent, 0);
//
//	cudaEventSynchronize(StopEvent);
//
//	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);
//
//	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);
//
//	cudaEventDestroy(StartEvent);
//	cudaEventDestroy(StopEvent);
//
//}


void Common_NeighborListCal_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerCommonGPU) {
	dim3 threads;
	dim3 blocks;
	int NB;
	int BlockNumEachBoxtemp;
	int BlockNumEachBox;

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox*NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	Kernel_NormalCalcNeighborList_multipleBox << < blocks, threads >> > (BlockNumEachBox,IDStartEnd_Dev, Dev_ClustersPosXYZ, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerCommonGPU, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);
}


void Common_NeighborListCal_CPU_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, double** Host_ClustersPosXYZ, int* Host_NNearestNeighbor) {

	double minDist;
	double Distance;
	for (int IBox = 0; IBox < NBox; IBox++) {
		for (int i = IDStartEnd_Host[IBox][0]; i <= IDStartEnd_Host[IBox][1]; i++) {

			minDist = 1.E16;

			for (int j = IDStartEnd_Host[IBox][0]; j <= IDStartEnd_Host[IBox][1]; j++) {
				if (i != j) {
					Distance = (Host_ClustersPosXYZ[i][0] - Host_ClustersPosXYZ[j][0])*(Host_ClustersPosXYZ[i][0] - Host_ClustersPosXYZ[j][0]) + \
						(Host_ClustersPosXYZ[i][1] - Host_ClustersPosXYZ[j][1])*(Host_ClustersPosXYZ[i][1] - Host_ClustersPosXYZ[j][1]) + \
						(Host_ClustersPosXYZ[i][2] - Host_ClustersPosXYZ[j][2])*(Host_ClustersPosXYZ[i][2] - Host_ClustersPosXYZ[j][2]);

					if (Distance < minDist) {
						minDist = Distance;
						Host_NNearestNeighbor[i] = j;
					}

				}
			}
		}

	}

}