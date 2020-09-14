#ifndef _MYNEIGHBORLIST_MULTIPLEBOX_
#define _MYNEIGHBORLIST_MULTIPLEBOX_

extern "C" void Common_NeighborListCal_multipleBox_Shared(int NClusters,int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Host_NNearestNeighbor, float &timerCommonGPU);

extern "C" void Common_NeighborListCal_multipleBox_noShared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Host_NNearestNeighbor, float &timerCommonGPU);


extern "C" void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_noShared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX,int* Host_NNearestNeighbor, float &timerMyMethod);
extern "C" void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_noShared_LeftRightCohen(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX, int* Host_NNearestNeighbor, float &timerMyMethod);
extern "C" void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_Shared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX,int* Host_NNearestNeighbor, float &timerMyMethod);

extern "C" void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_noShared_WithYLimit(int NClusters,
																						int NBox,
																						int **IDStartEnd_Host,
																						int **IDStartEnd_Dev,
																						double* ToSortDev_ClustersPosX,
																						double* ToSortDev_ClustersPosY,
																							double* ToSortDev_ClustersPosZ,
																						double** Dev_ClustersPosXYZ,
																						int* SortedIndexX,
																						int* SortedIndexY,
																						int* SortedIndexZ,
																						int* ReverseMap_SortedIndexX,
																						int* ReverseMap_SortedIndexY,
																						int* ReverseMap_SortedIndexZ,
																						int* Host_NNearestNeighbor,
																						float &timerMyMethod);



extern "C" void My_NeighborListCal_ArbitrayBitonicSortXY_multipleBox_noShared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double* ToSortDev_ClustersPosY, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* SortedIndexY, int* ReverseMap_SortedIndexX, int* ReverseMap_SortedIndexY, int* Host_NNearestNeighbor, float &timerMyMethod);

extern "C" void My_NeighborListCal_ArbitrayBitonicSortXY_multipleBox_Shared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double* ToSortDev_ClustersPosY, double** Dev_ClustersPosXYZ, int* SortedIndexX,  int* SortedIndexY, int* ReverseMap_SortedIndexX, int* ReverseMap_SortedIndexY, int* Host_NNearestNeighbor, float &timerMyMethod);

extern "C" void My_NeighborListCal_ArbitrayBitonicSortXYZ_multipleBox_noShared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double* ToSortDev_ClustersPosY, double* ToSortDev_ClustersPosZ, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* SortedIndexY, int* SortedIndexZ, int* Host_NNearestNeighbor, float &timerMyMethod);

extern "C" void My_NeighborListCal_ArbitrayBitonicSortXYZ_multipleBox_Shared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double* ToSortDev_ClustersPosY, double* ToSortDev_ClustersPosZ, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* SortedIndexY, int* SortedIndexZ,int* Host_NNearestNeighbor, float &timerMyMethod);


extern "C" void Common_NeighborListCal_CPU_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, double** Host_ClustersPosXYZ, int* Host_NNearestNeighbor);

#endif
