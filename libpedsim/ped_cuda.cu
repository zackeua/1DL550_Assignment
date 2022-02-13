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
	//static float *restrict mat_a __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
	cudaError_t cudaStatus;


    cudaMalloc((void**)&this->x, sizeof(float) * agents_array->agents.size());
    cudaMalloc((void**)&this->y, sizeof(float) * agents_array->agents.size());

	cudaMemcpy(this->x, agents_array->x, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->y, agents_array->y, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);

	
	cudaMalloc((void**)&this->dest_x, sizeof(float) * agents_array->agents.size());
    cudaMalloc((void**)&this->dest_y, sizeof(float) * agents_array->agents.size());
    cudaMalloc((void**)&this->dest_r, sizeof(float) * agents_array->agents.size());

	cudaMemcpy(this->dest_x, agents_array->dest_x, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->dest_y, agents_array->dest_y, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->dest_r, agents_array->dest_r, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);


	
    cudaMalloc((void**)&this->waypoint_ptr, sizeof(float) * agents_array->agents.size());
    cudaMalloc((void**)&this->waypoint_len, sizeof(float) * agents_array->agents.size());

	cudaMemcpy(this->waypoint_ptr, agents_array->waypoint_ptr, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(this->waypoint_len, agents_array->waypoint_len, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&this->waypoint_offset, sizeof(int) * agents_array->agents.size());

	

	

	int* waypoint_offset = new int[agents_array->agents.size()+1];

	waypoint_offset[0] = 0;
	int tot_length = 0;
	for (int i = 1; i <= agents_array->agents.size(); i++) {
		waypoint_offset[i] = waypoint_offset[i-1] + agents_array->waypoints[i-1]->size();
    }
	tot_length = waypoint_offset[agents_array->agents.size()];
	
	cudaMemcpy(this->waypoint_offset, waypoint_offset, sizeof(int) * (tot_length + 1), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&this->waypoint_x, sizeof(float) * tot_length);
    cudaMalloc((void**)&this->waypoint_y, sizeof(float) * tot_length);
    cudaMalloc((void**)&this->waypoint_r, sizeof(float) * tot_length);

	for (int i = 0; i < agents_array->agents.size(); i++) {

		cudaMemcpy(&this->waypoint_x[waypoint_offset[i]], agents_array->waypoint_x[i], sizeof(float) * agents_array->waypoint_len[i], cudaMemcpyHostToDevice);
		cudaMemcpy(&this->waypoint_y[waypoint_offset[i]], agents_array->waypoint_y[i], sizeof(float) * agents_array->waypoint_len[i], cudaMemcpyHostToDevice);
		cudaMemcpy(&this->waypoint_r[waypoint_offset[i]], agents_array->waypoint_r[i], sizeof(float) * agents_array->waypoint_len[i], cudaMemcpyHostToDevice);
	}
}

/*
void Ped::Cuagents::computeNextDesiredPosition(int i) {
	
	double diffX = dest_x[i] - this->x[i];
	double diffY = dest_y[i] - this->y[i];

	double len = sqrt(diffX * diffX + diffY * diffY);
	
	this->x[i] = (int)round(this->x[i] + diffX / len);
	this->y[i] = (int)round(this->y[i] + diffY / len);


	// If the destination is null, or if the agent has reached its destination, then we compute its new destination coordinates.
	if (len < this->dest_r[i]) {
		this->dest_x[i] = this->waypoint_x[i][this->waypoint_ptr[i]];
		this->dest_y[i] = this->waypoint_y[i][this->waypoint_ptr[i]];
		this->dest_r[i] = this->waypoint_r[i][this->waypoint_ptr[i]];

		this->waypoint_ptr[i] += 1;
		if (this->waypoint_ptr[i] == this->waypoint_len[i])
			this->waypoint_ptr[i] = 0;
	}



}

void Ped::Cuagents::free() {
	cudaFree((void**)&this->x);
	cudaFree((void**)&this->y);

	cudaFree((void**)&this->dest_x);
	cudaFree((void**)&this->dest_y);
	cudaFree((void**)&this->dest_r);
	
	for (int i = 0; i < this->size; i++) {

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

*/
