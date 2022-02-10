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

    this->x = new float[agents.size()];
    this->y = new float[agents.size()];
	this->dest_x = new float[agents.size()];
    this->dest_y = new float[agents.size()];
	this->dest_r = new float[agents.size()];

    this->destination = new Twaypoint*[agents.size()];
    this->lastDestination = new Twaypoint*[agents.size()];
    this->waypoints = new deque<Twaypoint*>*[agents.size()];

	
	for (int i = 0; i < agents.size(); i++) {
        this->x[i] = (float)agents[i]->getX();
        this->y[i] = (float)agents[i]->getY();
		
		this->destination[i] = agents[i]->destination;
		this->waypoints[i] = &(agents[i]->waypoints);
		this->lastDestination[i] = agents[i]->lastDestination;
    }
}


void Ped::Tagents::computeNextDesiredPosition(int i) {
	this->destination[i] = getNextDestination(i);
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
	this->x[i] = round(this->x[i] + diffX / len);
	this->y[i] = round(this->y[i] + diffY / len);

	this->agents[i]->setX((int)x[i]);
	this->agents[i]->setY((int)y[i]);
}
/*
void Ped::Tagents::addWaypoint(Twaypoint* wp, int i) {
	this->waypoints[i].push_back(wp);
}
*/
Ped::Twaypoint* Ped::Tagents::getNextDestination(int i) {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (this->destination[i] != NULL) {
		// compute if agent reached its current destination
		

		double diffX = dest_x[i] - this->x[i];
		double diffY = dest_y[i] - this->y[i];
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < this->dest_r[i];
		//std::cout << " " << this->x[i] << std::endl;
		//std::cout << " " << this->y[i] << std::endl;
		//std::cout << length << " " << this->destination[i]->getr() << std::endl;
	}

	if ((agentReachedDestination || this->destination[i] == NULL) && !this->waypoints[i]->empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		if (this->destination[i] != NULL) {
			this->waypoints[i]->push_back(this->destination[i]);
		}
		nextDestination = this->waypoints[i]->front();
		this->dest_x[i] = nextDestination->getx();
		this->dest_y[i] = nextDestination->gety();
		this->dest_r[i] = nextDestination->getr();

		this->waypoints[i]->pop_front();
		// DO NOT print destination here, might be NULL
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = this->destination[i];
	}

	return nextDestination;
}
