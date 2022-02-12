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

    this->x = new float[agents.size()]; __attribute__((aligned(32)))
    this->y = new float[agents.size()]; __attribute__((aligned(32)))
	this->dest_x = new float[agents.size()]; __attribute__((aligned(32)))
    this->dest_y = new float[agents.size()]; __attribute__((aligned(32)))
	this->dest_r = new float[agents.size()]; __attribute__((aligned(32)))

    this->destination = new Twaypoint*[agents.size()];
    this->lastDestination = new Twaypoint*[agents.size()];
    this->waypoints = new deque<Twaypoint*>*[agents.size()];
	this->agentReachedDestination = false;

	this->waypoint_x = new float*[agents.size()];
	this->waypoint_y = new float*[agents.size()];
	this->waypoint_r = new float*[agents.size()];

	this->waypoint_ptr = new int[agents.size()];
	this->waypoint_len = new int[agents.size()];
	
	for (int i = 0; i < agents.size(); i++) {
        this->x[i] = (float)agents[i]->getX();
        this->y[i] = (float)agents[i]->getY();
		
		this->destination[i] = agents[i]->destination;
		this->waypoints[i] = &(agents[i]->waypoints);
		this->lastDestination[i] = agents[i]->lastDestination;

		this->waypoint_x[i] = new float[this->waypoints[i]->size()];
		this->waypoint_y[i] = new float[this->waypoints[i]->size()];
		this->waypoint_r[i] = new float[this->waypoints[i]->size()];
		this->waypoint_ptr[i] = 0;
		this->waypoint_len[i] = this->waypoints[i]->size();

		for (int j = 0; j < this->waypoints[i]->size(); j++) {
			this->waypoint_x[i][j] = this->waypoints[i]->at(j)->getx();
			this->waypoint_y[i][j] = this->waypoints[i]->at(j)->gety();
			this->waypoint_r[i][j] = this->waypoints[i]->at(j)->getr();
		}
    }
}

void Ped::Tagents::computeNextDesiredPosition(int i) {
	updateNextDestination(i);
	this->destination[i] = this->agentReachedDestination || this->destination[i] == NULL ?\
						   this->waypoints[i]->front() : this->destination[i];
	
	if (this->destination[i] == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}
	// Safe to print here
	//std::cout << this->destination[i]->getx() << std::endl;	
	//std::cout << this->destination[i]->gety() << std::endl;

	//double diffX = destination[i]->getx() - this->x[i];
	//double diffY = destination[i]->gety() - this->y[i];

	// SIMD: recleare diffX and diffY as simd
	double diffX = dest_x[i] - this->x[i];
	double diffY = dest_y[i] - this->y[i];

	double len = sqrt(diffX * diffX + diffY * diffY);
	
	// SIMD:
	this->x[i] = (int)round(this->x[i] + diffX / len);
	this->y[i] = (int)round(this->y[i] + diffY / len);

	this->agents[i]->setX(x[i]);
	this->agents[i]->setY(y[i]);
}

void Ped::Tagents::updateNextDestination(int i) {
	// If the destination isn't null, then compute if the agent reached its destination and add it to the end of the deque.
	if (this->destination[i] != NULL) {
		double diffX = dest_x[i] - this->x[i];
		double diffY = dest_y[i] - this->y[i];
		double length = sqrt(diffX * diffX + diffY * diffY);
		this->agentReachedDestination = length < this->dest_r[i];
		this->waypoints[i]->push_back(this->destination[i]);
	}

	// If the destination is null, or if the agent has reached its destination, then we compute its new destination coordinates.
	if (this->destination[i] == NULL || this->agentReachedDestination) {
		this->dest_x[i] = this->waypoint_x[i][this->waypoint_ptr[i]];
		this->dest_y[i] = this->waypoint_y[i][this->waypoint_ptr[i]];
		this->dest_r[i] = this->waypoint_r[i][this->waypoint_ptr[i]];

		this->waypoint_ptr[i] += 1;
		if (this->waypoint_ptr[i] == this->waypoint_len[i])
			this->waypoint_ptr[i] = 0;
	}
}
