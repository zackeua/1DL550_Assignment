//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_cuda.h"
#include "ped_waypoint.h"
#include <math.h>

#include <stdlib.h>
#include <iostream>

Ped::Cuagents::Cuagents(Ped::Tagents* agents_array) {
	// The CUDA error status needed to check if the allocations and copies succeeded
	cudaError_t cudaStatus;

	// Allocating the agent coordinates on the processor
    cudaMalloc((void**)&this->x, sizeof(float) * agents_array->agents.size());
    cudaMalloc((void**)&this->y, sizeof(float) * agents_array->agents.size());

	// Copying them to the graphics card
	cudaMemcpy(this->x, agents_array->x, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->y, agents_array->y, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);

	// Allocating the agent destinations on the processor
	cudaMalloc((void**)&this->dest_x, sizeof(float) * agents_array->agents.size());
    cudaMalloc((void**)&this->dest_y, sizeof(float) * agents_array->agents.size());
    cudaMalloc((void**)&this->dest_r, sizeof(float) * agents_array->agents.size());

	// Copying them to the graphics card
	cudaMemcpy(this->dest_x, agents_array->dest_x, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->dest_y, agents_array->dest_y, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->dest_r, agents_array->dest_r, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);

	// Allocating the waypoint pointer and the waypoint length on the processor
    cudaMalloc((void**)&this->waypoint_ptr, sizeof(float) * agents_array->agents.size());
    cudaMalloc((void**)&this->waypoint_len, sizeof(float) * agents_array->agents.size());

	// Copying them to the graphics card
	cudaMemcpy(this->waypoint_ptr, agents_array->waypoint_ptr, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->waypoint_len, agents_array->waypoint_len, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);

	// Allocating the waypoint offset vector needed for indexing the squeezed-together array
	cudaMalloc((void**)&this->waypoint_offset, sizeof(int) * (agents_array->agents.size()+1));

	// Allocating the waypoint offset on the processor
	int* waypoint_offset = new int[agents_array->agents.size()+1];

	// Setting the current offset and setting it based on the actual offsets
	waypoint_offset[0] = 0;
	int tot_length = 0;
	for (int i = 1; i <= agents_array->agents.size(); i++)
		waypoint_offset[i] = waypoint_offset[i-1] + agents_array->waypoints[i-1]->size();
    
	// Computing the total offset length
	tot_length = waypoint_offset[agents_array->agents.size()];
	
	// Copying it to the graphics card
	cudaMemcpy(this->waypoint_offset, waypoint_offset, sizeof(int) * (tot_length), cudaMemcpyHostToDevice);
	
	// Allocating the waypoint coordinates on the processor
	cudaMalloc((void**)&this->waypoint_x, sizeof(float) * tot_length);
    cudaMalloc((void**)&this->waypoint_y, sizeof(float) * tot_length);
    cudaMalloc((void**)&this->waypoint_r, sizeof(float) * tot_length);

	// Copying them to the graphics card
	for (int i = 0; i < agents_array->agents.size(); i++) {
		cudaMemcpy(&this->waypoint_x[waypoint_offset[i]], agents_array->waypoint_x[i], sizeof(float) * agents_array->waypoint_len[i], cudaMemcpyHostToDevice);
		cudaMemcpy(&this->waypoint_y[waypoint_offset[i]], agents_array->waypoint_y[i], sizeof(float) * agents_array->waypoint_len[i], cudaMemcpyHostToDevice);
		cudaMemcpy(&this->waypoint_r[waypoint_offset[i]], agents_array->waypoint_r[i], sizeof(float) * agents_array->waypoint_len[i], cudaMemcpyHostToDevice);
	}
}

// Freeing the allocated memory
void Ped::Cuagents::free(Ped::Tagents* agents_array) {
	cudaFree((void**)&this->x);
	cudaFree((void**)&this->y);

	cudaFree((void**)&this->dest_x);
	cudaFree((void**)&this->dest_y);
	cudaFree((void**)&this->dest_r);
	
	for (int i = 0; i < agents_array->agents.size(); i++) {
		cudaFree((void**)&this->waypoint_x[i]);
		cudaFree((void**)&this->waypoint_y[i]);
		cudaFree((void**)&this->waypoint_r[i]);
    }

	cudaFree((void**)&this->waypoint_x);
	cudaFree((void**)&this->waypoint_y);
	cudaFree((void**)&this->waypoint_r);
	cudaFree((void**)&this->waypoint_ptr);
	cudaFree((void**)&this->waypoint_len);
}

