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
    this->x = new int[agents.size()];
    this->y = new int[agents.size()];

	this->dest_x = new int[agents.size()];
    this->dest_y = new int[agents.size()];
	this->dest_r = new int[agents.size()];

    this->destination = new Twaypoint*[agents.size()];
    this->lastDestination = new Twaypoint*[agents.size()];
    this->waypoints = new deque<Twaypoint*>*[agents.size()];
	
	for (int i = 0; i < agents.size(); i++) {
        this->x[i] = agents[i]->getX();
        this->y[i] = agents[i]->getY();
		
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
	// this->destination is safe to print below

	// SIMD
	double diffX = dest_x[i] - this->x[i];
	double diffY = dest_y[i] - this->y[i];

	double len = sqrt(diffX * diffX + diffY * diffY);
	
	// SIMD
	this->x[i] = (int)round(this->x[i] + diffX / len);
	this->y[i] = (int)round(this->y[i] + diffY / len);
}

Ped::Twaypoint* Ped::Tagents::getNextDestination(int i) {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (this->destination[i] != NULL) {
		// Compute if agent reached its current destination
		double diffX = dest_x[i] - this->x[i];
		double diffY = dest_y[i] - this->y[i];
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < this->dest_r[i];
	}

	if ((agentReachedDestination || this->destination[i] == NULL) && !this->waypoints[i]->empty()) {
		// Case 1: Agent has reached destination (or has no current destination);
		// get next destination if available
		if (this->destination[i] != NULL) {
			this->waypoints[i]->push_back(this->destination[i]);
		}
		nextDestination = this->waypoints[i]->front();
		this->dest_x[i] = nextDestination->getx();
		this->dest_y[i] = nextDestination->gety();
		this->dest_r[i] = nextDestination->getr();

		this->waypoints[i]->pop_front();
		// Do not print this->destination here, since it might be NULL
	}
	else {
		// Case 2: Agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = this->destination[i];
	}

	return nextDestination;
}
