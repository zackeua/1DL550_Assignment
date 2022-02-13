//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_cuda.h"
#include <math.h>

#include <stdlib.h>
#include <iostream>

Ped::Cuagents::Cuagents(Ped::Tagents* agents_array) {
	//static float *restrict mat_a __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
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

	
	for (int i = 0; i < agents_array->agents.size(); i++) {
        this->x[i] = (float)agents_array->agents[i]->getX();
        this->y[i] = (float)agents_array->agents[i]->getY();
		

        cudaMallocManaged((void**)&this->waypoint_x[i], sizeof(float) * agents_array->waypoints[i]->size());
        cudaMallocManaged((void**)&this->waypoint_y[i], sizeof(float) * agents_array->waypoints[i]->size());
        cudaMallocManaged((void**)&this->waypoint_r[i], sizeof(float) * agents_array->waypoints[i]->size());
		this->waypoint_ptr[i] = 0;
		this->waypoint_len[i] = this->waypoints[i]->size();

		for (int j = 0; j < this->waypoints[i]->size(); j++) {
			this->waypoint_x[i][j] = this->waypoints[i]->at(j)->getx();
			this->waypoint_y[i][j] = this->waypoints[i]->at(j)->gety();
			this->waypoint_r[i][j] = this->waypoints[i]->at(j)->getr();
		}
    }
}

__device__
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

