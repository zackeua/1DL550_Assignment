// Created for Low Level Parallel Programming 2017.
// Implements the heatmap functionality. 

#include "ped_model.h"
#include "heatmap_cuda.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// Allocating the memory on the GPU.
void Ped::Model::setupHeatmapCUDA() {
	cudaMalloc((void**)&heatmap_cuda, sizeof(int) * SIZE * SIZE);
	cudaMalloc((void**)&scaled_heatmap_cuda, sizeof(int) * SCALED_SIZE * SCALED_SIZE);
	cudaMalloc((void**)&blurred_heatmap_cuda, sizeof(int) * SCALED_SIZE * SCALED_SIZE);
}

// Fading the heatmap.
__global__ void fadeHeatmapCUDA(int* heatmap) {
	// Looping over every one thread in each block.
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x)
		heatmap[i] = (int)round(heatmap[i] * 0.80);
}

// Incrementing the heatmap where there are agents.
__global__ void incrementHeatCUDA(int numberOfAgents, int* heatmap, float* desiredX, float* desiredY) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfAgents; i += blockDim.x * gridDim.x) {
		int x = desiredX[i];
		int y = desiredY[i];

		// If the desired location is outside the box, then we jump to the next iteration.
		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
			continue;

		// Using atomic addition since multiple i may have same y and x.
		atomicAdd(&heatmap[y * SIZE + x], 40);
	}

}

// Capping the heatmap values to the RGB maximum.
__global__ void capHeatmapCUDA(int* heatmap) {
	for (int i =  blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x)
		heatmap[i] = heatmap[i] < 255 ? heatmap[i] : 255;
}

// Scaling the heatmap for visual representation.
__global__ void scaledHeatmapCUDA(int* heatmap, int* scaled_heatmap) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x) {
		// Obtaining the pixel value that we want to give the agent.
		int value = heatmap[i];

		// Looping over the pixels in the cell to paint them the same value.
		for (int cellY = 0; cellY < CELLSIZE; cellY++)
			for (int cellX = 0; cellX < CELLSIZE; cellX++) {
							// Mapping the one-dimensional coordinate to x and y coordinates.
				scaled_heatmap[(i % SIZE) * CELLSIZE + cellX + // x
				               (i / SIZE) * SCALED_SIZE * CELLSIZE + cellY * SCALED_SIZE] = value; // y
			}
	}
}

// Blurring the heatmap for visual representation.
__global__ void blurredHeatmapCUDA(int* scaled_heatmap, int* blurred_cuda) {
	// Weights for the Gaussian blur filter.
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};
	
	// We use this sum to compute how much to blur the cell.
	#define WEIGHTSUM 273

	// This is two lines down plus two columns in because of the filter size.
	// We use this offset in order to fit the filter correctly.
	#define OFFSET SCALED_SIZE * 2 + 2
	
	// This is gridDim.x here. It has to be specified here for CUDA to work.
	const int num_block = 80;

	// These are the lengths of the sides of the block.
	const int side_length = SCALED_SIZE / num_block;
	
	// These are the padded block side lengths.
	const int padded_side_length = side_length + 4;
	
	// This is the size of the padded block.
	const int padded_heatmap_size = padded_side_length * padded_side_length;

	// Allocating a heatmap for the block using shared memory for about 10x higher speed.
	__shared__ int shared_heatmap[padded_heatmap_size];

	// Jumping blockIdx.x * side_length to the right, blockIdx.y * side_length * SCALED_SIZE down, and
	// two rows up and two columns left as - 2 * SCALED_SIZE - 2.
	int offset = blockIdx.x * side_length + blockIdx.y * side_length * SCALED_SIZE - 2 * SCALED_SIZE - 2;

	// Only copying over the information in scaled_heatmap to shared_heatmap using the first thread in each block.
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (int y = 0; y < padded_side_length; y++) {
			for (int x = 0; x < padded_side_length; x++) {
				// If the desired location is outside the box, then we copy the values.
				if (0 <= y * SCALED_SIZE + x + offset && y * SCALED_SIZE + x + offset < SCALED_SIZE * SCALED_SIZE) {
					shared_heatmap[y * padded_side_length + x] = scaled_heatmap[y * SCALED_SIZE + x + offset];
				} else { // Otherwise, we simply add zero.
					shared_heatmap[y * padded_side_length + x] = 0;
				}
			}
		}
	}

	// Ensuring that the heatmap is copied everywhere.
	__syncthreads();

	// Blurring the results using a Gaussian blur filter.
	for	(int y = threadIdx.y; y < side_length; y += blockDim.y) {
		for	(int x = threadIdx.x; x < side_length; x += blockDim.x) {
			// Translating the two-dimensional coordinates to a one-dimensional coordinate
			int idx = y * SCALED_SIZE + x + offset;

			// If the desired location is outside the box, then we jump to the next iteration
			if (idx < SCALED_SIZE * 2 || idx > SCALED_SIZE * (SCALED_SIZE - 2) || idx % SCALED_SIZE < 2 || idx % SCALED_SIZE > SCALED_SIZE - 2)
				continue;
			
			int sum = 0;
			for (int k = -2; k < 3; k++)
				for (int l = -2; l < 3; l++) // Obtaining the right value by adding an offset to the locally remapped coordinates from 2D to 1D.
					sum += w[2 + k][2 + l] * shared_heatmap[2 + 2 * padded_side_length + y * padded_side_length + x + l + k * padded_side_length];

			// Computing the average value to render
			int value = sum / WEIGHTSUM;

			// Using bitwise or in order to render the values which 
			blurred_cuda[y * SCALED_SIZE + x + offset] = 0x00FF0000 | value << 24;
		}
	}
}

// Updating the heatmap according to the agent positions
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

	// Copying over the desiredX and desiredY of the agents from the processor to the graphics card.
	cudaMemcpy(this->cuda_array.desiredX, this->agents_array->desiredX, this->agents.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->cuda_array.desiredY, this->agents_array->desiredY, this->agents.size() * sizeof(float), cudaMemcpyHostToDevice);

	// Fading from the last step.
	fadeHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (this->heatmap_cuda);

	// Incrementing in this step.
	incrementHeatCUDA <<<number_of_blocks, threads_per_block>>> (this->agents.size(), this->heatmap_cuda, this->cuda_array.desiredX, this->cuda_array.desiredY);

	// Capping the values to avoid exceeding the maximum of RGB.
	capHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (this->heatmap_cuda);

	// Scaling the heatmap in order to render the agents and not one pixel.
	scaledHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (this->heatmap_cuda, this->scaled_heatmap_cuda);
	
	// Blurring the heatmap.
	dim3 grid(80, 80);
	dim3 block(10, 10);
	blurredHeatmapCUDA <<<grid, block>>> (this->scaled_heatmap_cuda, this->blurred_heatmap_cuda);

	// Copying back the results.
	cudaMemcpy(this->blurred_heatmap[0], this->blurred_heatmap_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}
