#include<iostream>
#include<math.h>
#include<cuda_runtime.h>
#include"device_launch_parameters.h" 
#include<fstream>
#define MAX_NUM_LISTS 512

using namespace std;
int num_lists = 256; // the number of parallel threads

__device__ void radix_sort_Sample(double* const data_0, double* const data_1, int num_lists, int num_data, int tid);
__device__ void merge_list_Sample(const double* src_data, double* const dest_list, int num_lists, int num_data, int tid);
__device__ void preprocess_double_Sample(double* const data, int num_lists, int num_data, int tid);
__device__ void Aeprocess_double_Sample(double* const data, int num_lists, int num_data, int tid);

__global__ void GPU_radix_sort_Sample(double* const src_data, double* const dest_data, int num_lists, int num_data)
{
	// temp_data:temporarily store the data
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	// special preprocessing of IEEE floating-point numbers before applying radix sort
	preprocess_double_Sample(src_data, num_lists, num_data, tid);
	__syncthreads();
	// no shared memory
	//radix_sort_Sample(src_data, dest_data, num_lists, num_data, tid);
	__syncthreads();
	//merge_list_Sample(src_data, dest_data, num_lists, num_data, tid);
	__syncthreads();
	Aeprocess_double_Sample(dest_data, num_lists, num_data, tid);
	__syncthreads();
}

__device__ void preprocess_double_Sample(double* const src_data, int num_lists, int num_data, int tid)
{
	for (int i = tid; i < num_data; i += num_lists)
	{
		unsigned long long int *data_temp = (unsigned long long int*)(&src_data[i]);

		*data_temp = ((*data_temp >> 63) & 0x1) ? ~(*data_temp) : (*data_temp) | 0x8000000000000000;
	}
}

__device__ void Aeprocess_double_Sample(double* const data, int num_lists, int num_data, int tid)
{
	for (int i = tid; i < num_data; i += num_lists)
	{
		unsigned long long int* data_temp = (unsigned long long int*)(&data[i]);
		*data_temp = (*data_temp >> 63 & 0x1) ? (*data_temp) & 0x7fffffffffffffff : ~(*data_temp);
	}
}


__device__ void radix_sort_Sample(double* const data_0, double* const data_1, int num_lists, int num_data, int tid)
{
	for (int bit = 0; bit < 64; bit++)
	{
		unsigned long long int bit_mask = (1llu << bit);
		int count_0 = 0;
		int count_1 = 0;
		for (int i = tid; i < num_data; i += num_lists)
		{
			unsigned long long int*temp = (unsigned long long int*)&data_0[i];

			if (*temp & bit_mask)
			{
				data_1[tid + count_1 * num_lists] = data_0[i]; //bug 在这里 等于时会做强制类型转化
				count_1 += 1;
			}
			else {
				data_0[tid + count_0 * num_lists] = data_0[i];
				count_0 += 1;
			}
		}
		for (int j = 0; j < count_1; j++)
		{
			data_0[tid + count_0 * num_lists + j * num_lists] = data_1[tid + j * num_lists];
		}
	}
}

__device__ void merge_list_Sample(const double* src_data, double* const dest_list, int num_lists, int num_data, int tid)
{
	int num_per_list = ceil((double)num_data / num_lists);
	__shared__ int list_index[MAX_NUM_LISTS];
	__shared__ double record_val[MAX_NUM_LISTS];
	__shared__ int record_tid[MAX_NUM_LISTS];
	list_index[tid] = 0;
	record_val[tid] = 0;
	record_tid[tid] = tid;
	__syncthreads();
	for (int i = 0; i < num_data; i++)
	{
		record_val[tid] = 0;
		record_tid[tid] = tid; // bug2 每次都要进行初始化
		if (list_index[tid] < num_per_list)
		{
			int src_index = tid + list_index[tid] * num_lists;
			if (src_index < num_data)
			{
				record_val[tid] = src_data[src_index];
			}
			else {
				unsigned long long int*temp = (unsigned long long int*)&record_val[tid];
				*temp = 0xffffffffffffffff;
			}
		}
		else {
			unsigned long long int*temp = (unsigned long long int*)&record_val[tid];
			*temp = 0xffffffffffffffff;
		}
		__syncthreads();
		int tid_max = num_lists >> 1;
		while (tid_max != 0)
		{
			if (tid < tid_max)
			{
				unsigned long long int* temp1 = (unsigned long long int*)&record_val[tid];
				unsigned long long int*temp2 = (unsigned long long int*)&record_val[tid + tid_max];
				if (*temp2 < *temp1)
				{
					record_val[tid] = record_val[tid + tid_max];
					record_tid[tid] = record_tid[tid + tid_max];
				}
			}
			tid_max = tid_max >> 1;
			__syncthreads();
		}
		if (tid == 0)
		{
			list_index[record_tid[0]]++;
			dest_list[i] = record_val[0];
		}
		__syncthreads();
	}
}
extern "C" void radixSort_Sample(int NSize, double* Host_TestArrayIn, double* Host_TestArrayOut, int dir, float & timerRadixSort)
{
	double *Dev_Data_Src;
	double  *Dev_Data_Dest;

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);



	cudaMalloc((void**)&Dev_Data_Src, sizeof(double)*NSize);
	cudaMalloc((void**)&Dev_Data_Dest, sizeof(double)*NSize);
	cudaMemcpy(Dev_Data_Src, Host_TestArrayIn, sizeof(double)*NSize, cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();

	cudaEventRecord(StartEvent, 0);




	GPU_radix_sort_Sample << <1, num_lists >> > (Dev_Data_Src, Dev_Data_Dest, num_lists, NSize);

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerRadixSort, StartEvent, StopEvent);

	cudaDeviceSynchronize();

	cudaMemcpy(Host_TestArrayOut, Dev_Data_Dest, sizeof(double)*NSize, cudaMemcpyDeviceToHost);

	cudaFree(Dev_Data_Src);
	cudaFree(Dev_Data_Dest);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);
}


int main_wait()
{
	int num_data = 1000;
	double *data = new double[num_data];
	double  *src_data, *dest_data;
	for (int i = 0; i < num_data; i++)
	{
		data[i] = (double)rand() / double(RAND_MAX);
	}
	cudaMalloc((void**)&src_data, sizeof(double)*num_data);
	cudaMalloc((void**)&dest_data, sizeof(double)*num_data);
	cudaMemcpy(src_data, data, sizeof(double)*num_data, cudaMemcpyHostToDevice);

	GPU_radix_sort_Sample << <1, num_lists >> > (src_data, dest_data, num_lists, num_data);
	cudaMemcpy(data, dest_data, sizeof(double)*num_data, cudaMemcpyDeviceToHost);

	for (int i = 0; i < num_data; i++)
	{
		cout << data[i] << " ";
	}

	return 0;
}
