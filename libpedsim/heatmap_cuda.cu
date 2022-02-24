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

// Sets up the heatmap
void Ped::Model::setupHeatmapCUDA() {
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	for (int i = 0; i < SCALED_SIZE; i++) {
		blurred_heatmap[i] = bhm + SCALED_SIZE*i;
	}

	// Allocationg memory on GPU
	cudaMalloc((void**)&heatmap, sizeof(int) * SIZE * SIZE);
	cudaMalloc((void**)&scaled_heatmap, sizeof(int) * SCALED_SIZE * SCALED_SIZE);
	cudaMalloc((void**)&blurred_cuda, sizeof(int) * SCALED_SIZE * SCALED_SIZE);
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
	}
}

__global__ void capHeatmapCUDA(int* heatmap) {
	for (int i =  blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x)
		heatmap[i] = heatmap[i] < 255 ? heatmap[i] : 255;
}

__global__ void scaledHeatmapCUDA(int* heatmap, int* scaled_heatmap) {
	// Scale the data for visual representation
	for (int i =  blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x) {
		int value = heatmap[i];
		for (int cellY = 0; cellY < CELLSIZE; cellY++)
			for (int cellX = 0; cellX < CELLSIZE; cellX++)
				scaled_heatmap[(i % SIZE) * CELLSIZE + cellX +
				               (i / SIZE) * SCALED_SIZE + cellY * SCALED_SIZE] = value;
	}
}

__global__ void blurredHeatmapCUDA(int* blurred_cuda, int* scaled_heatmap) {
	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

	#define WEIGHTSUM 273
	#define OFFSET SCALED_SIZE * 2 + 2
	// Apply Gaussian blurfilter
	for (int i = 0; i < (SCALED_SIZE - 2) * (SCALED_SIZE - 2); i++) {
		int sum = 0;
		for (int k = -2; k < 3; k++)
			for (int l = -2; l < 3; l++)
				sum += w[2 + k][2 + l] * scaled_heatmap[OFFSET + i % (SCALED_SIZE - 2) + l + k * SCALED_SIZE];

		int value = sum / WEIGHTSUM;
		blurred_cuda[i] = 0x00FF0000 | value << 24;
	}
}

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapCUDA() {
	// Setting the number of threads
	int number_of_blocks = 100;
	int threads_per_block = 100;
	
	fadeHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (*this->heatmap);

	incrementHeatCUDA <<<number_of_blocks, threads_per_block>>> (this->agents.size(), *this->heatmap, this->cuda_array.desiredX, this->cuda_array.desiredY);

	capHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (*this->heatmap);
	
	scaledHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (*this->heatmap, *this->scaled_heatmap);
	
	blurredHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (*this->blurred_cuda, *this->scaled_heatmap);

	cudaMemcpy(this->blurred_heatmap, this->blurred_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}
