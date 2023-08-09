#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include<string>
#include<vector>
#include<float.h>
#include<ctime>
#include<iomanip>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <queue>
#include <list>
#include <tuple>
#include <chrono>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cublas_v2.h>

// For units of time such as h, min, s, ms, us, ns
using namespace std::literals::chrono_literals;
#define M 10000
#define K 2
#define N 10000
#define nc 2

long long  int Axy = M * K;
long long  int Bxy = K * N;
long long  int Cxy = M * N;
const int unassigned = -1;
const float infinity_PDC = std::numeric_limits<float>::infinity();
// printf("%d\n",Cxy);
using namespace std::chrono;

//初始化，二维矩阵一维化
void initial(float *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (float)(i + 1);
	}
}
//转置矩阵
void Tranpose2D(float *input, float *output)
{
	int temp = 0;
	for(int i =0; i < K;i++)
		for(int j = 0; j < M; j++)
			{output[temp] = input[K*j+i];
			temp +=1;}
}
//打印二维矩阵

void printMatrix(float *array, int row, int col)
{
	float *p = array;
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			printf("%5f ", p[x]);
		}
		p = p + col;
		printf("\n");
	}
	printf("------------------------------------------------------------------------------------\n");
	return;
}
void printMatrixInt(int *array, int row, int col)
{
	int *p = array;
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			printf("%5d ", p[x]);
		}
		p = p + col;
		printf("\n");
	}
	printf("------------------------------------------------------------------------------------\n");
	return;
}
/************************************************************cuda算法********************************************************************/
//欧氏距离计算，优化思路，矩阵具有对称性，因此很多计算量可以减半
__global__ void L2computeOnDevice(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;//row number
	int iy = threadIdx.y + blockDim.y*blockIdx.y;//col number

	if (ix < N_p && iy < M_p)
	{
		float sum = 0;
		for (int k = 0; k < K_p; k++)
		{
			sum += powf(array_A[iy*K_p + k] - array_B[k*N_p + ix],2);//sum += sqrt(pow(array_A[iy*K_p + k] - array_B[k*N_p + ix],2));
		}
		array_C[iy*N_p + ix] = sqrt(sum);
	}
}

//曼哈顿距离：无size版本
__global__ void CitycomputeOnDevice(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;//row number
	int iy = threadIdx.y + blockDim.y*blockIdx.y;//col number
	if (ix < N_p && iy < M_p)
	{
		float sum = 0;
		for (int k = 0; k < K_p; k++)
		{
			sum += abs(array_A[iy*K_p + k] - array_B[k*N_p + ix]);
		}
		array_C[iy*N_p + ix] = sum;
	}
}
__global__ void InitDistanceAsc(int *array_A)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;//row number
	int iy = threadIdx.y + blockDim.y*blockIdx.y;//col number
	array_A[iy*M+ix] = ix;
}

__global__ void SortDistanceAsc(int *array_A, float *array_C)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;//row number
	int iy = threadIdx.y + blockDim.y*blockIdx.y;//col number
	if(array_A[iy*M+ix] == 0)
	{
		thrust::sort(array_A + iy*M+ix, array_A + iy*M+ix + M, [&](int a, int b) { return array_C[iy*M+ix + a] < array_C[iy*M+ix + b]; });
	}
}

__global__ void ComputerRHO(float *array_A,float *array_B)
{
	__shared__ float similarityDesc[M];
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int i = blockId * (blockDim.x * blockDim.y) +(threadIdx.y * blockDim.x) + threadIdx.x;
	if(i % M == 0 and i < M*M)
	{
		// float *similarityDesc = array_A + i;
		thrust::copy(thrust::seq,array_A + i, array_A + i +  M, similarityDesc);
		
		thrust::sort(similarityDesc, similarityDesc + M, thrust::greater<int >());
		// for (int j = 0 ;j < M;j++) {printf("%f ",similarityDesc[j]);printf("\n");}
		array_B[i/M] = thrust::reduce(thrust::seq,similarityDesc, similarityDesc + K, 0);
	}

}

int main( )
{
	cudaEvent_t gpustart, gpustop;

	cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL*1024);

	// long long  int nBytes = M * N * sizeof(float);
	float *h_A = (float*)malloc(Axy * sizeof(float));
	float *h_B = (float*)malloc(Bxy * sizeof(float));
	float *deviceRef = (float*)malloc(Cxy * sizeof(float));

	// const auto pathData = "/home/ivyadmin/Project_demo/repeter_Means/result_1.tsv";
	// int label[M];
	// const auto fileData = fopen(pathData, "r");
	// for (int i = 0; i < M; i++)
	// {
	// 	int temp = fscanf(fileData, "%f %f %d\n", &h_A[i * K], &h_A[i * K + 1], &label[i]);
	// 	if(temp == -1) printf("%d,error\n",temp);
	// }
	// fclose(fileData);


	initial(h_A, Axy);
	// printMatrix(h_A,M,K);
	// const auto time = high_resolution_clock::now();
	float elapsedTime = 0.0;
	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);
	printf("-------------------------------------Start-----------------------------------------------\n");
	
	//-------------------------------------归一化-------------------------------------
	
	float least[K], most[K];
	const auto infinity = std::numeric_limits<float>::infinity();
	std::fill(least, least + K, infinity);
	std::fill(most, most + K, -infinity);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < K; j++) {
			least[j] = thrust::min(least[j], h_A[i * K + j]);
			most[j] = thrust::max(most[j], h_A[i * K + j]);
		}
	for (int i = 0; i < M; i++)
		for (int j = 0; j < K; j++)
			h_A[i * K + j] = (h_A[i * K + j] - least[j]) / (most[j] - least[j]);

	Tranpose2D(h_A, h_B);

	float *d_A, *d_B, *distance;
	cudaMalloc((void**)&d_A, Axy/10 * sizeof(float));
	cudaMalloc((void**)&d_B, Bxy/10 * sizeof(float));
	cudaMalloc((void**)&distance, Cxy * sizeof(float));
	cudaMemcpy(d_A, h_A, Axy/10 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bxy/10 * sizeof(float), cudaMemcpyHostToDevice);


	int dimx = 2;
	int dimy = 2;
	dim3 block(dimx, dimy);
	dim3 grid((M + block.x - 1) / block.x*M, (N + block.y - 1) / block.y*M);


	
	//-------------------------------------计算欧式距离-------------------------------------0.21s
	CitycomputeOnDevice<<<grid,block>>> (d_A, d_B, distance, M, K, N);//L2computeOnDevice、 CitycomputeOnDevice
   cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
	
	cudaMemcpy(deviceRef, distance, Cxy * sizeof(float), cudaMemcpyDeviceToHost);

	
	//-------------------------------------计算neighor-------------------------------------0.34s
	// int *indexDistanceAsc = (int*)malloc(Cxy * sizeof(int));
	// int *indexNeighbor = (int*)malloc(Axy * sizeof(int));
	// int *d_indexDistanceAsc, *d_indexNeighbor;
	// cudaMalloc((void**)&d_indexDistanceAsc, Cxy * sizeof(int));
	// cudaMalloc((void**)&d_indexNeighbor, Axy * sizeof(int));

	// InitDistanceAsc<<<grid,block>>>(d_indexDistanceAsc);
	// SortDistanceAsc<<<grid,block>>>(d_indexDistanceAsc,distance);
	// cudaMemcpy(indexDistanceAsc, d_indexDistanceAsc, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
	// #pragma omp parallel for//------------------------------并行加速
	// for (int i = 0; i < M; i++) {
	// 	std::copy(indexDistanceAsc + i * M, indexDistanceAsc + i * M + K, indexNeighbor + i * K);
	// 	std::sort(indexNeighbor + i * K, indexNeighbor + (i + 1) * K); // For set_intersection()
	// }

	//-------------------------------------计算shared neighor-------------------------------------0.62s
	// int *indexSharedNeighbor = (int*)malloc(M * M * K * sizeof(long long int));
	// int *numSharedNeighbor = (int*)malloc(M * M  * sizeof(int));
	// #pragma omp parallel for//------------------------------并行加速
	// for (int i = 0; i < M; i++) {
	// 		numSharedNeighbor[i * M + i] = 0;
	// 	for (int j = 0; j < i; j++) {
	// 		numSharedNeighbor[j * M + i] = numSharedNeighbor[i * M + j] = std::set_intersection(
	// 			indexNeighbor + i * K, indexNeighbor + (i + 1) * K,
	// 			indexNeighbor + j * K, indexNeighbor + (j + 1) * K,
	// 			indexSharedNeighbor + i * M * K + j * K
	// 		) - (indexSharedNeighbor + i * M * K + j * K);
	// 		std::copy(
	// 			indexSharedNeighbor + i * M * K + j * K,
	// 			indexSharedNeighbor + i * M * K + j * K + numSharedNeighbor[i * M + j],
	// 			indexSharedNeighbor + j * M * K + i * K
	// 		);
	// 	}}
	// 	const auto similarity = new float[M * M];
	// 	std::fill(similarity, similarity + M * M, 0);

	// 	#pragma omp parallel for//------------------------------并行加速
	// 	for (int i = 0; i < M; i++) {
	// 		similarity[i * M + i] = 0;
	// 		for (int j = 0; j < i; j++) {
	// 			const auto first = indexSharedNeighbor + i * M * K + j * K;
	// 			const auto last = indexSharedNeighbor + i * M * K + j * K + numSharedNeighbor[i * M + j];
	// 			if (std::binary_search(first, last, i) && std::binary_search(first, last, j)) {
	// 				float sum = 0;
	// 				for (int u = 0; u < numSharedNeighbor[i * M + j]; u++)
	// 				{
	// 					const int shared = indexSharedNeighbor[i * M * K + j * K + u];
	// 					sum += deviceRef[i * M + shared] + deviceRef[j * M + shared];
	// 					// printf("%d\n",i * M + shared);
	// 				}
	// 				// printf("%f\n",sum);
	// 				similarity[j * M + i] = similarity[i * M + j] = pow(numSharedNeighbor[i * M + j], 2) / sum;
	// 			}
	// 		}
	// 	}
	// // 	
	
	// delete[] indexSharedNeighbor;

	//-------------------------------------计算ρ-------------------------------------cpu:0.5S
	// const auto rho = new float[M];
	// const auto similarityDesc = new float[M];
	// float *d_similarity, *d_rho;
	// cudaMalloc((void**)&d_rho, M * sizeof(float));
	// cudaMalloc((void**)&d_similarity, M *M* sizeof(float));
	// cudaMemcpy(d_similarity, similarity, M *M* sizeof(float), cudaMemcpyHostToDevice);
	// ComputerRHO<<<grid,block>>>(d_similarity,d_rho);
	// cudaMemcpy(rho, d_rho, M * sizeof(float), cudaMemcpyDeviceToHost);
	// delete[] similarity;
	// delete[] similarityDesc;
	// printMatrix(rho, 1, M);

	// -------------------------------------Compute δ-------------------------------------------
	// const auto delta = new float[M];
	// const auto distanceNeighborSum = new float[M];
	// const auto indexRhoDesc = new int[M];
	// std::fill(delta, delta + M, infinity_PDC);
	// std::fill(distanceNeighborSum, distanceNeighborSum + M, 0);
	
	// for (int i = 0; i < M; i++)
	// 	for (int j = 0; j < K; j++)
	// 		distanceNeighborSum[i] += deviceRef[i * M + indexNeighbor[i * K + j]];
	// std::iota(indexRhoDesc, indexRhoDesc + M, 0);
	// std::sort(indexRhoDesc, indexRhoDesc + M, [&](int a, int b) { return rho[a] > rho[b]; });

	// #pragma omp parallel for//------------------------------并行加速
	// for (int i = 1; i < M; i++) {
	// 	int a = indexRhoDesc[i];
	// 	for (int j = 0; j < i; j++) {
	// 		int b = indexRhoDesc[j];
	// 		delta[a] = std::min(delta[a], deviceRef[a * M + b] * (distanceNeighborSum[a] + distanceNeighborSum[b]));
	// 	}
	// }
	// delta[indexRhoDesc[0]] = -infinity_PDC;
	// delta[indexRhoDesc[0]] = *std::max_element(delta, delta + M);
	// delete[] deviceRef;
	// delete[] distanceNeighborSum;
	// delete[] indexRhoDesc;
	

	// ---------------------------------Compute γ-----------------------------------------------
	// const auto gamma = new float[M];
	// for (int i = 0; i < M; i++)
	// 	gamma[i] = rho[i] * delta[i];
	// delete[] rho;
	// delete[] delta;	
	
	// ------------------------------------Compute centroid--------------------------------------------

	// const auto indexAssignment = new int[M];
	// const auto indexCentroid = new int[nc];
	// const auto indexGammaDesc = new int[M];
	// std::fill(indexAssignment, indexAssignment + M, unassigned);
	// std::iota(indexGammaDesc, indexGammaDesc + M, 0);
	// thrust::sort(indexGammaDesc, indexGammaDesc + M, [&](int a, int b) { return gamma[a] > gamma[b]; });
	// thrust::copy(indexGammaDesc, indexGammaDesc + nc, indexCentroid);
	// thrust::sort(indexCentroid, indexCentroid + nc);
	// for (int i = 0; i < nc; i++)
	// 	indexAssignment[indexCentroid[i]] = i;
	// delete[] gamma;
	// delete[] indexGammaDesc;

	// --------------------------------Assign non centroid step 1------------------------------------------------
	// std::queue<int> queue;
	// for (int i = 0; i < nc; i++)
	// 	queue.push(indexCentroid[i]);
	// while (!queue.empty()) {
	// 	int a = queue.front();
	// 	queue.pop();
	// 	for (int i = 0; i < K; i++) {
	// 		int b = indexNeighbor[a * K + i];
	// 		if (indexAssignment[b] == unassigned && numSharedNeighbor[a * M + b] * 2 >= K) {
	// 			indexAssignment[b] = indexAssignment[a];
	// 			queue.push(b);
	// 		}
	// 	}
	// }
	// delete[] indexNeighbor;
	// delete[] numSharedNeighbor;
	// -----------------------------------Assign non centroid step 2---------------------------------------------

	// int tempK = K;
	// std::list<int> indexUnassigned;
	// for (int i = 0; i < M; i++)
	// 	if (indexAssignment[i] == unassigned)
	// 		indexUnassigned.push_back(i);

	// int numUnassigned = indexUnassigned.size();
	// const auto numNeighborAssignment = new int[numUnassigned * nc];
	// while (numUnassigned) {
	// 	std::fill(numNeighborAssignment, numNeighborAssignment + numUnassigned * nc, 0);
	// 	int i = 0;
	// 	for (const auto& a : indexUnassigned) {
	// 		for (int j = 0; j < tempK; j++) {
	// 			int b = indexDistanceAsc[a * M + j];
	// 			if (indexAssignment[b] != unassigned)
	// 				++numNeighborAssignment[i * nc + indexAssignment[b]];
	// 		}
	// 		i++;
	// 	}
	// 	if (int most = *std::max_element(numNeighborAssignment, numNeighborAssignment + numUnassigned * nc)) {
	// 		auto it = indexUnassigned.begin();
	// 		for (int j = 0; j < numUnassigned; j++) {
	// 			const auto first = numNeighborAssignment + j * nc;
	// 			const auto last = numNeighborAssignment + (j + 1) * nc;
	// 			const auto current = std::find(first, last, most); // In MATLAB, if multiple hits, the last will be used
	// 			if (current == last) ++it;
	// 			else {
	// 				indexAssignment[*it] = current - first;
	// 				it = indexUnassigned.erase(it);
	// 			}
	// 		}
	// 		numUnassigned = indexUnassigned.size();
	// 	}
	// 	else tempK++;
	// }
	// delete[] indexDistanceAsc;
	// delete[] numNeighborAssignment;
	// printMatrixInt(indexCentroid, 1, nc);
	// printMatrixInt(indexAssignment, 1, M);

	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);
	cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);


	printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fms\n",
			M, N, grid.x, grid.y, block.x, block.y, elapsedTime);
	// printf("Time Cost = %ldms\n", duration_cast<milliseconds>(high_resolution_clock::now() - time).count());
	// cudaError_t err = cudaGetLastError();
	// if (err != cudaSuccess) {
	// 	printf("CUDA Error: %s\n", cudaGetErrorString(err));
	// 	// Possibly: exit(-1) if program cannot continue....
	// }
	printf("------------------------------------------------------------------------------------\n");
	// printMatrix(deviceRef, M, N);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(distance);

	free(h_A);
	free(h_B);
	// free(deviceRef);

	cudaDeviceReset();

	return (0);
}

