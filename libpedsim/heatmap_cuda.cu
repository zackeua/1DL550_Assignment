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
		// need atomic because multiple i may have same y and x
		atomicAdd(&heatmap[y * SIZE + x], 40);
		//heatmap[y * SIZE + x] += 40;
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
	// printf("gridDim.x: %d\ngridDim.y: %d\nblockIdx.x: %d\nblockIdx.y: %d\nblockDim.x: %d\nblockDim.y: %d\nthreadIdx.x: %d\nthreadIdx.y: %d\n\n\n",
	// 		gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);

	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

	#define WEIGHTSUM 273
	#define OFFSET SCALED_SIZE * 2 + 2
	



	// int block = blockIdx.x;
	const int num_block = 80;

	const int shared_heatmap_size = SCALED_SIZE*SCALED_SIZE/num_block/num_block;
	const int padded_heatmap_size = (SCALED_SIZE/num_block + 4)*(SCALED_SIZE/num_block + 4);
									//SCALED_SIZE*SCALED_SIZE/num_block/num_block + ;
	const int padded_side_length = SCALED_SIZE/num_block + 4;
	const int side_length = SCALED_SIZE/num_block;					

	__shared__ int heatmap[padded_heatmap_size];

	int offset =  blockIdx.x * side_length + blockIdx.y * side_length * SCALED_SIZE - 2 - 2 * SCALED_SIZE;

	for (int y = 0; y < padded_side_length; y++) {
		for (int x = 0; x < padded_side_length; x++) {
			if (0 <= y*SCALED_SIZE + x + offset && y*SCALED_SIZE + x + offset < SCALED_SIZE * SCALED_SIZE)
				heatmap[y * padded_side_length + x] = scaled_heatmap[y*SCALED_SIZE + x + offset];
		}
	}



	for	(int y = threadIdx.y; y < padded_side_length; y += blockDim.y) {
		for	(int x = threadIdx.x; x < padded_side_length; x += blockDim.x) { // loop over pixels in local grid
			int idx = y*SCALED_SIZE + x + offset;

			if (idx < SCALED_SIZE * 2 || idx > SCALED_SIZE * (SCALED_SIZE-2) || idx % SCALED_SIZE < 2 || idx % SCALED_SIZE > (SCALED_SIZE - 2))
				continue;

			int sum = 0;
			for (int k = -2; k < 3; k++)
				for (int l = -2; l < 3; l++) { // apply kernel
					
					//sum += w[2 + k][2 + l] * heatmap[y * padded_side_length + x + l + k * padded_side_length];
					sum += w[2 + k][2 + l] * scaled_heatmap[y * SCALED_SIZE + x + l + k * SCALED_SIZE];
					printf("%d", scaled_heatmap[y * SCALED_SIZE + x + l + k * SCALED_SIZE] - heatmap[y * padded_side_length + x + l + k * padded_side_length]);
				}		
			int value = sum / WEIGHTSUM;
			blurred_cuda[idx] = 0x00FF0000 | value << 24;
		}
	}



	// 	// Apply Gaussian blurfilter
	// for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < padded_side_length * padded_side_length; i += blockDim.y * blockDim.x) {
	// 	int idx = i;
		
	// 	if (idx < SCALED_SIZE * 2 || idx > SCALED_SIZE * (SCALED_SIZE-2) || idx % SCALED_SIZE < 2 || idx % SCALED_SIZE > (SCALED_SIZE - 2)) {
	// 		//printf("kancka\n");
	// 		continue;
	// 	}
	// 	// else {
	// 	// 	printf("pelle\n");
	// 	// }
		
	// 	int sum = 0;
	// 	for (int k = -2; k < 3; k++)
	// 		for (int l = -2; l < 3; l++) {
				
	// 			sum += w[2 + k][2 + l] * heatmap[i + l + k * padded_side_length];
	// 			//sum += w[2 + k][2 + l] * scaled_heatmap[i + l + k * SCALED_SIZE];
	// 		}


	// 	int value = sum / WEIGHTSUM;
	// 	blurred_cuda[i + offset] = 0xFFFF0000; // | value << 24;
	// }
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

	dim3 grid(80, 80);
	dim3 block(1, 1);
	blurredHeatmapCUDA <<<grid, block>>> (this->scaled_heatmap_cuda, this->blurred_heatmap_cuda);

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
