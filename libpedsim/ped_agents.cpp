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

Ped::Tagents::Tagents(std::vector<Ped::Tagent*> agents) {
    x = new int[agents.size()];
    y = new int[agents.size()];
    dest = new Twaypoint*[agents.size()];
    lastDest = new Twaypoint*[agents.size()];
    waypts = new deque<Twaypoint*>[agents.size()];

    return; // Debug
    for (int i = 0; i < agents.size(); i++) {
        x[i] = agents[i]->getX();
        y[i] = agents[i]->getY();
        dest[i] = agents[i]->dest;
        waypts[i] = agents[i]->waypts;
        lastDest[i] = agents[i]->lastDest;
    }
}

void Ped::Tagents::computeNextDesiredPosition(int i) {
	this->dest[i] = getNextDestination(i);
	if (this->dest[i] == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}

	double diffX = dest[i]->getx() - this->x[i];
	double diffY = dest[i]->gety() - this->y[i];
	double len = sqrt(diffX * diffX + diffY * diffY);
	this->x[i] = (int)round(this->x[i] + diffX / len);
	this->y[i] = (int)round(this->y[i] + diffY / len);
}

Ped::Twaypoint* Ped::Tagents::getNextDestination(int i) {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (this->dest != NULL) {
		// compute if agent reached its current destination
		double diffX = this->dest[i]->getx() - this->x[i];
		double diffY = this->dest[i]->gety() - this->y[i];
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < this->dest[i]->getr();
	}

	if ((agentReachedDestination || this->dest[i] == NULL) && !(this->waypts[i].empty())) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		this->waypts[i].push_back(this->dest[i]);
		nextDestination = this->waypts[i].front();
		this->waypts[i].pop_front();
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = this->dest[i];
	}

	return nextDestination;
}
