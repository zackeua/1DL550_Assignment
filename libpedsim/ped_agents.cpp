//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_agents.h"
#include "ped_waypoint.h"
#include <math.h>

#include <stdlib.h>
#include <iostream>

Ped::Tagents::Tagents(std::vector<Ped::Tagent*> agents) {
	//static float *restrict mat_a __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

	this->agents = agents;

    this->x = new float[agents.size()];// __attribute__((aligned(32)))
    this->y = new float[agents.size()];// __attribute__((aligned(32)))
	this->dest_x = new float[agents.size()];// __attribute__((aligned(32)))
    this->dest_y = new float[agents.size()];// __attribute__((aligned(32)))
	this->dest_r = new float[agents.size()];// __attribute__((aligned(32)))

    this->waypoints = new deque<Twaypoint*>*[agents.size()];

	this->waypoint_x = new float*[agents.size()];
	this->waypoint_y = new float*[agents.size()];
	this->waypoint_r = new float*[agents.size()];

	this->waypoint_ptr = new int[agents.size()];
	this->waypoint_len = new int[agents.size()];
	
	for (int i = 0; i < agents.size(); i++) {
        this->x[i] = (float)agents[i]->getX();
        this->y[i] = (float)agents[i]->getY();
		
		this->waypoints[i] = &(agents[i]->waypoints);

		this->waypoint_x[i] = new float[agents[i]->waypoints.size()];
		this->waypoint_y[i] = new float[agents[i]->waypoints.size()];
		this->waypoint_r[i] = new float[agents[i]->waypoints.size()];
		this->waypoint_ptr[i] = 0;
		this->waypoint_len[i] = agents[i]->waypoints.size();

		for (int j = 0; j < agents[i]->waypoints.size(); j++) {
			this->waypoint_x[i][j] = agents[i]->waypoints.at(j)->getx();
			this->waypoint_y[i][j] = agents[i]->waypoints.at(j)->gety();
			this->waypoint_r[i][j] = agents[i]->waypoints.at(j)->getr();
		}
		this->dest_x[i] = this->waypoint_x[i][this->waypoint_ptr[i]];
		this->dest_y[i] = this->waypoint_y[i][this->waypoint_ptr[i]];
		this->dest_r[i] = this->waypoint_r[i][this->waypoint_ptr[i]];
		this->waypoint_ptr[i] = 1;
    }


}


/*void Ped::Tagents::computeNextDesiredPosition2(int i) {
	updateDestination(i);
	this->destination[i] = this->agentReachedDestination || this->destination[i] == NULL ?\
						   this->waypoints[i]->front() : this->destination[i];
	
	if (this->destination[i] == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}
	// Safe to print here

	double diffX = dest_x[i] - this->x[i];
	double diffY = dest_y[i] - this->y[i];

	double len = sqrt(diffX * diffX + diffY * diffY);
	
	this->x[i] = (int)round(this->x[i] + diffX / len);
	this->y[i] = (int)round(this->y[i] + diffY / len);

	this->agents[i]->setX(x[i]);
	this->agents[i]->setY(y[i]);
}

void Ped::Tagents::updateDestination2(int i) {
	// This might help when vectorizing:
	// https://stackoverflow.com/questions/38006616/how-to-use-if-condition-in-intrinsics
	// https://community.intel.com/t5/Intel-C-Compiler/use-of-if-else-statement-in-sse2-intrinsics/td-p/816362

	// If the destination isn't null, then compute if the agent reached its destination and add it to the end of the deque.
	if (this->destination[i] != NULL) {
		double diffX = dest_x[i] - this->x[i];
		double diffY = dest_y[i] - this->y[i];
		double length = sqrt(diffX * diffX + diffY * diffY);
		this->agentReachedDestination[i] = length < this->dest_r[i];
	}

	// If the destination is null, or if the agent has reached its destination, then we compute its new destination coordinates.
	if (this->destination[i] == NULL || this->agentReachedDestination[i]) {
		this->dest_x[i] = this->waypoint_x[i][this->waypoint_ptr[i]];
		this->dest_y[i] = this->waypoint_y[i][this->waypoint_ptr[i]];
		this->dest_r[i] = this->waypoint_r[i][this->waypoint_ptr[i]];

		this->waypoint_ptr[i] += 1;
		if (this->waypoint_ptr[i] == this->waypoint_len[i])
			this->waypoint_ptr[i] = 0;
	}
}
*/











void Ped::Tagents::computeNextDesiredPosition(int i) {
	
	// Safe to print here

	double diffX = dest_x[i] - this->x[i];
	double diffY = dest_y[i] - this->y[i];

	double len = sqrt(diffX * diffX + diffY * diffY);
	
	this->x[i] = (int)round(this->x[i] + diffX / len);
	this->y[i] = (int)round(this->y[i] + diffY / len);

	this->agents[i]->setX(x[i]);
	this->agents[i]->setY(y[i]);

	// If the destination isn't null, then compute if the agent reached its destination and add it to the end of the deque.
	
	diffX = dest_x[i] - this->x[i];
	diffY = dest_y[i] - this->y[i];
	len = sqrt(diffX * diffX + diffY * diffY);
	

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
