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
	this->size = agents_array->agents.size();

	//cudaError_t cudaStatus;


    cudaMallocManaged((void**)&this->x, sizeof(float) * agents_array->agents.size());
    cudaMallocManaged((void**)&this->y, sizeof(float) * agents_array->agents.size());

    cudaMallocManaged((void**)&this->dest_x, sizeof(float) * agents_array->agents.size());
    cudaMallocManaged((void**)&this->dest_y, sizeof(float) * agents_array->agents.size());
    cudaMallocManaged((void**)&this->dest_r, sizeof(float) * agents_array->agents.size());

    cudaMallocManaged((void**)&this->waypoint_x, sizeof(float*) * agents_array->agents.size());
    cudaMallocManaged((void**)&this->waypoint_y, sizeof(float*) * agents_array->agents.size());
    cudaMallocManaged((void**)&this->waypoint_r, sizeof(float*) * agents_array->agents.size());
    cudaMallocManaged((void**)&this->waypoint_ptr, sizeof(float) * agents_array->agents.size());
    cudaMallocManaged((void**)&this->waypoint_len, sizeof(float) * agents_array->agents.size());

	cudaMemcpy((void**)&this->x, (void**)&agents_array->x, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy((void**)&this->y, (void**)&agents_array->y, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);

	cudaMemcpy((void**)&this->waypoint_ptr, (void**)&agents_array->waypoint_ptr, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);
	cudaMemcpy((void**)&this->waypoint_len, (void**)&agents_array->waypoint_len, sizeof(float) * agents_array->agents.size(), cudaMemcpyHostToDevice);



	for (int i = 0; i < agents_array->agents.size(); i++) {

        cudaMallocManaged((void**)&this->waypoint_x[i], sizeof(float) * agents_array->waypoints[i]->size());
        cudaMallocManaged((void**)&this->waypoint_y[i], sizeof(float) * agents_array->waypoints[i]->size());
        cudaMallocManaged((void**)&this->waypoint_r[i], sizeof(float) * agents_array->waypoints[i]->size());

		cudaMemcpy((void**)&this->waypoint_x[i], (void**)&agents_array->waypoint_x[i], sizeof(float) * agents_array->waypoints[i]->size(), cudaMemcpyHostToDevice);
		cudaMemcpy((void**)&this->waypoint_y[i], (void**)&agents_array->waypoint_y[i], sizeof(float) * agents_array->waypoints[i]->size(), cudaMemcpyHostToDevice);
		cudaMemcpy((void**)&this->waypoint_r[i], (void**)&agents_array->waypoint_r[i], sizeof(float) * agents_array->waypoints[i]->size(), cudaMemcpyHostToDevice);
    }
}


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

