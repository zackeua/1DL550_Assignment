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
    destination = new Twaypoint*[agents.size()]; // plocka ut dess x och y från denna så man kan ladda in flera sånna x och y med SIMD
    lastDestination = new Twaypoint*[agents.size()];


    waypoints = new deque<Twaypoint*>[agents.size()];
	return; // debug exit
	for (int i = 0; i < agents.size(); i++) {
        x[i] = agents[i]->getX();
        y[i] = agents[i]->getY();
		destination[i] = agents[i]->destination;
		waypoints[i] = agents[i]->waypoints;
		lastDestination[i] = agents[i]->lastDestination;
    }
}


void Ped::Tagents::computeNextDesiredPosition(int i) {
	this->destination[i] = getNextDestination(i);
	if (this->destination[i] == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}

	double diffX = destination[i]->getx() - this->x[i];
	double diffY = destination[i]->gety() - this->y[i];
	double len = sqrt(diffX * diffX + diffY * diffY);
	this->x[i] = (int)round(this->x[i] + diffX / len);
	this->y[i] = (int)round(this->y[i] + diffY / len);
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
		double diffX = this->destination[i]->getx() - this->x[i];
		double diffY = this->destination[i]->gety() - this->y[i];
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < this->destination[i]->getr();
	}

	if ((agentReachedDestination || this->destination[i] == NULL) && !this->waypoints[i].empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		this->waypoints[i].push_back(this->destination[i]);
		nextDestination = this->waypoints[i].front();
		this->waypoints[i].pop_front();
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = this->destination[i];
	}

	return nextDestination;
}
