// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"
#include  "heatmap_cuda.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// Allocationg memory on GPU
void Ped::Model::setupHeatmapCUDA() {
	cudaMalloc((void**)&heatmap_cuda, sizeof(int) * SIZE * SIZE);
	cudaMalloc((void**)&scaled_heatmap_cuda, sizeof(int) * SCALED_SIZE * SCALED_SIZE);
	cudaMalloc((void**)&blurred_heatmap_cuda, sizeof(int) * SCALED_SIZE * SCALED_SIZE);
}

__global__ void fadeHeatmapCUDA(int* heatmap) {
	for (int i =  blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x)
		heatmap[i] = (int)round(heatmap[i] * 0.80);
}

__global__ void incrementHeatCUDA(int numberOfAgents, int* heatmap, float* desiredX, float* desiredY) {
	// Count how many agents want to go to each location
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfAgents; i += blockDim.x * gridDim.x) {
		int x = desiredX[i];
		int y = desiredY[i];

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
			continue;

		// intensify heat for better color results
		heatmap[y * SIZE + x] += 40;
		// printf("Heatmap[%f * SIZE + %f]: %d\n",desiredX[i], desiredY[i],  heatmap[y * SIZE + x]);
	}

}

__global__ void capHeatmapCUDA(int* heatmap) {
	for (int i =  blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x)
		heatmap[i] = heatmap[i] < 255 ? heatmap[i] : 255;
}

__global__ void scaledHeatmapCUDA(int* heatmap, int* scaled_heatmap) {
	// Scale the data for visual representation
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x) {
		int value = heatmap[i];
		for (int cellY = 0; cellY < CELLSIZE; cellY++)
			for (int cellX = 0; cellX < CELLSIZE; cellX++) {

				// scaled_heatmap[i + CELLSIZE * cellX + CELLSIZE * cellY] = value;
				scaled_heatmap[(i % SIZE) * CELLSIZE + cellX +
				                (i / SIZE) * SCALED_SIZE * CELLSIZE + cellY * SCALED_SIZE] = value;
			}
		// if (value != 0)
		// 	printf("scaled_heatmap[%d] = %d\n", i, value);
	}
}

__global__ void blurredHeatmapCUDA(int* scaled_heatmap, int* blurred_cuda) {
	//Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

	#define WEIGHTSUM 273
	#define OFFSET SCALED_SIZE * 2 + 2
	// // Apply Gaussian blurfilter
	// for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (SCALED_SIZE -2) * (SCALED_SIZE-2); i += blockDim.x * gridDim.x) {
	// // for (int i = 0; i < (SCALED_SIZE - 2) * (SCALED_SIZE - 2); i++) {
	// 	int sum = 0;
	// 	for (int k = -2; k < 3; k++)
	// 		for (int l = -2; l < 3; l++)
	// 			sum += w[2 + k][2 + l] * scaled_heatmap[OFFSET + i % (SCALED_SIZE - 2) + l + k * SCALED_SIZE];


	// 	int value = sum / WEIGHTSUM;
	// 	blurred_cuda[i] = 0x00FF0000 | value << 24;
	// }
	// for (int block = blockIdx.x; block < 2; block += 1) {
	// 	int* shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	// 	__shared__ int** heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

	// 	for (int i = 0; i < SCALED_SIZE; i++)
	// 	{
	// 		heatmap[i] = shm + SCALED_SIZE*i;
	// 	}

	int block = blockIdx.x;

	__shared__ int heatmap[SCALED_SIZE*SCALED_SIZE/blockDim.x + SCALED_SIZE*4 +4];
	for (int i = 0; i < SCALED_SIZE*SCALED_SIZE/blockDim.x + SCALED_SIZE*4 +4; i++) {
		int offset = -2 * SCALED_SIZE;
		if (block == 0) {
			offset = 0;
		} else if (block == blockDim.x -1) {
			offset = -4 * SCALED_SIZE;
		}
		heatmap[i] = scaled_heatmap[i +SCALED_SIZE*SCALED_SIZE/blockDim.x * block + offset];
	}



		// Apply Gaussian blurfilter
	for (int i = blockDim.x + threadIdx.x; i < (SCALED_SIZE) * (SCALED_SIZE); i += blockDim.x) {
		if (i < SCALED_SIZE * 2 || i > SCALED_SIZE * (SCALED_SIZE-2) || i % SCALED_SIZE < 2 || i % SCALED_SIZE > (SCALED_SIZE - 2))
			continue;
		
		int sum = 0;
		for (int k = -2; k < 3; k++)
			for (int l = -2; l < 3; l++) {
				int offset = 2 * SCALED_SIZE;
				if (block == 0) {
					offset = 0;
				} else if (block == blockDim.x -1) {
					offset = 4 * SCALED_SIZE;
				}
				sum += w[2 + k][2 + l] * heatmap[offset + i + l + k * SCALED_SIZE];
				//sum += w[2 + k][2 + l] * scaled_heatmap[i + l + k * SCALED_SIZE];
			}


		int value = sum / WEIGHTSUM;
		blurred_cuda[i] = 0x00FF0000 | value << 24;
	}
	// }

}

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapCUDA() {
	// Setting the number of threads
	int number_of_blocks = 10;
	int threads_per_block = 10;
	
	// Allocating the CUDA status
	cudaError_t cudaStatus;

	// Setting the CUDA device
	cudaStatus = cudaSetDevice(0);

	// Checking if that worked
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice0 failed!\n");
		return;
	}

	fadeHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (this->heatmap_cuda);

	// // Synchronizing the threads
	// cudaStatus = cudaDeviceSynchronize();

	// // Checking if that worked
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaSetDevice1 failed!\n");
	// 	fprintf(stderr, "%d\n", cudaStatus);
	// 	return;
	// }

	cudaMemcpy(this->cuda_array.desiredX, this->agents_array->desiredX, this->agents.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->cuda_array.desiredY, this->agents_array->desiredY, this->agents.size() * sizeof(float), cudaMemcpyHostToDevice);

	incrementHeatCUDA <<<number_of_blocks, threads_per_block>>> (this->agents.size(), this->heatmap_cuda, this->cuda_array.desiredX, this->cuda_array.desiredY);

	// // Synchronizing the threads
	// cudaStatus = cudaDeviceSynchronize();

	// // Checking if that worked
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaSetDevice2 failed!\n");
	// 	fprintf(stderr, "%d\n", cudaStatus);
		
	// 	return;
	// }

	capHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (this->heatmap_cuda);
	
	// // Synchronizing the threads
	// cudaStatus = cudaDeviceSynchronize();

	// // Checking if that worked
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaSetDevice3 failed!\n");
	// 	fprintf(stderr, "%d\n", cudaStatus);
		
	// 	return;
	// }

	scaledHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (this->heatmap_cuda, this->scaled_heatmap_cuda);
	
	// // Synchronizing the threads
	// cudaStatus = cudaDeviceSynchronize();
	
	// // Checking if that worked
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaSetDevice4 failed!\n");
	// 	fprintf(stderr, "%d\n", cudaStatus);
		
	// 	return;
	// }

	blurredHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (this->scaled_heatmap_cuda, this->blurred_heatmap_cuda);

	// // Synchronizing the threads
	// cudaStatus = cudaDeviceSynchronize();
	
	// // Checking if that worked
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaSetDevice5 failed!\n");
	// 	fprintf(stderr, "%d\n", cudaStatus);
		
	// 	return;
	// }

	// cudaMemcpy(this->blurred_heatmap[0], this->scaled_heatmap_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(this->blurred_heatmap[0], this->blurred_heatmap_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	int i;
	std::cin >> i;

}
