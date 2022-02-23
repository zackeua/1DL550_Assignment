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
void Ped::Model::setupHeatmapCUDA()
{
	// Allocationg memory on GPU
	cudaMalloc((void**)&heatmap, sizeof(int) * SIZE * SIZE);
	cudaMalloc((void**)&scaled_heatmap, sizeof(int) * SCALED_SIZE * SCALED_SIZE);
	cudaMalloc((void**)&blurred_heatmap, sizeof(int) * SCALED_SIZE * SCALED_SIZE);
}

__global__ void fadeHeatmapCUDA(int* heatmap) {
	for (int i =  blockIdx.x * blockDim.x + threadIdx.x; i < SIZE * SIZE; i += blockDim.x * gridDim.x) {
		heatmap[i] = (int)round(heatmap[i] * 0.80);
	}
}

__global__ void incrementHeatCUDA(int numberOfAgents, int* heatmap, float* desiredX, float* desiredY) {
	//Count how many agents want to go to each location
	for (int i = 0; i < numberOfAgents; i++) {
		int x = desiredX[i];
		int y = desiredY[i];

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		{
			continue;
		}

		// intensify heat for better color results
		heatmap[y * SIZE + x] += 40;

	}

}


// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapCUDA()
{

	// Setting the number of threads
	int number_of_blocks = 100;
	int threads_per_block = 100;
	

	fadeHeatmapCUDA <<<number_of_blocks, threads_per_block>>> (*this->heatmap);


	incrementHeatCUDA <<<number_of_blocks, threads_per_block>>> (this->agents.size(), *this->heatmap, this->cuda_array->desiredX, this->cuda_array->desiredY);


// 	for (int x = 0; x < SIZE; x++)
// 	{
// 		for (int y = 0; y < SIZE; y++)
// 		{
// 			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
// 		}
// 	}

// 	// Scale the data for visual representation
// 	for (int y = 0; y < SIZE; y++)
// 	{
// 		for (int x = 0; x < SIZE; x++)
// 		{
// 			int value = heatmap[y][x];
// 			for (int cellY = 0; cellY < CELLSIZE; cellY++)
// 			{
// 				for (int cellX = 0; cellX < CELLSIZE; cellX++)
// 				{
// 					scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
// 				}
// 			}
// 		}
// 	}

// 	// Weights for blur filter
// 	const int w[5][5] = {
// 		{ 1, 4, 7, 4, 1 },
// 		{ 4, 16, 26, 16, 4 },
// 		{ 7, 26, 41, 26, 7 },
// 		{ 4, 16, 26, 16, 4 },
// 		{ 1, 4, 7, 4, 1 }
// 	};

// #define WEIGHTSUM 273
// 	// Apply gaussian blurfilter		       
// 	for (int i = 2; i < SCALED_SIZE - 2; i++)
// 	{
// 		for (int j = 2; j < SCALED_SIZE - 2; j++)
// 		{
// 			int sum = 0;
// 			for (int k = -2; k < 3; k++)
// 			{
// 				for (int l = -2; l < 3; l++)
// 				{
// 					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
// 				}
// 			}
// 			int value = sum / WEIGHTSUM;
// 			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
// 		}
// 	}
}
