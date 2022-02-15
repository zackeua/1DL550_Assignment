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
	
	this->agents = agents;

	// Setting the agent coordinates and the destination coordinates, and acceptance radius
    this->x = new float[agents.size()];// __attribute__((aligned(32)))
    this->y = new float[agents.size()];// __attribute__((aligned(32)))
	this->desiredX = new float[agents.size()];// __attribute__((aligned(32)))
    this->desiredY = new float[agents.size()];// __attribute__((aligned(32)))
	this->dest_x = new float[agents.size()];// __attribute__((aligned(32)))
    this->dest_y = new float[agents.size()];// __attribute__((aligned(32)))
	this->dest_r = new float[agents.size()];// __attribute__((aligned(32)))

	// Allocating the waypoints, as they will be copied to the respective waypoint coordinate arrays
    this->waypoints = new deque<Twaypoint*>*[agents.size()];

	// Allocating the waypoint coordinate and acceptance radius arrays
	this->waypoint_x = new float*[agents.size()];
	this->waypoint_y = new float*[agents.size()];
	this->waypoint_r = new float*[agents.size()];

	// Allocating the waypoint pointer which keeps track of which the next waypoint is for each agent
	this->waypoint_ptr = new int[agents.size()];
	this->waypoint_len = new int[agents.size()];
	
	// Filling the agent coordiantes, allocating the waypoints in the waypoint arrays, filling them, and filling the first destinations from thre.
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

		// We do this in order to ensure that the next destination never is null, since that only and always happens in the first timestep
		this->dest_x[i] = this->waypoint_x[i][this->waypoint_ptr[i]];
		this->dest_y[i] = this->waypoint_y[i][this->waypoint_ptr[i]];
		this->dest_r[i] = this->waypoint_r[i][this->waypoint_ptr[i]];
		this->waypoint_ptr[i] = 1;
    }
}

///////////////////////////////////////////////////////////////////////////

void Ped::Tagents::computeNextDesiredPosition(int i) {
	// Computing the difference from the current location and the destination coordinatewise
	double diffX = dest_x[i] - this->x[i];
	double diffY = dest_y[i] - this->y[i];

	// Computing the length to the destination based on the coordinatewise differences
	double len = sqrt(diffX * diffX + diffY * diffY);
	
	// Updating the new position from the differnces in x and y divided by the length
	this->x[i] = (int)round(this->x[i] + diffX / len);
	this->y[i] = (int)round(this->y[i] + diffY / len);

	// Updating the agents to reflect these changes in the graphics
	this->agents[i]->setX(x[i]);
	this->agents[i]->setY(y[i]);

	// Computing the new distance to the destination to check if we are there yet
	diffX = dest_x[i] - this->x[i];
	diffY = dest_y[i] - this->y[i];
	len = sqrt(diffX * diffX + diffY * diffY);

	// Making the comparison to see if we need to get the next waypoint
	if (len < this->dest_r[i]) {
		this->dest_x[i] = this->waypoint_x[i][this->waypoint_ptr[i]];
		this->dest_y[i] = this->waypoint_y[i][this->waypoint_ptr[i]];
		this->dest_r[i] = this->waypoint_r[i][this->waypoint_ptr[i]];

		this->waypoint_ptr[i] += 1;
		if (this->waypoint_ptr[i] == this->waypoint_len[i])
			this->waypoint_ptr[i] = 0;
	}
}

///////////////////////////////////////////////////////////////////////////

void Ped::Tagents::computeNextDesiredPositionMove(int i) {
	// Computing the difference from the current location and the destination coordinatewise
	double diffX = dest_x[i] - this->x[i];
	double diffY = dest_y[i] - this->y[i];

	// Computing the length to the destination based on the coordinatewise differences
	double len = sqrt(diffX * diffX + diffY * diffY);
	
	// Updating the new position from the differnces in x and y divided by the length
	this->desiredX[i] = (int)round(this->x[i] + diffX / len);
	this->desiredY[i] = (int)round(this->y[i] + diffY / len);

	// Updating the agents to reflect these changes in the graphics
	this->agents[i]->setDesiredX(desiredX[i]);
	this->agents[i]->setDesiredY(desiredY[i]);

	this->agents[i]->desiredPositionX = desiredX[i];
	this->agents[i]->desiredPositionY = desiredY[i];
}

void Ped::Tagents::reachedDestination(int i) {
	// Computing the new distance to the destination to check if we are there yet
	float diffX = dest_x[i] - this->x[i];
	float diffY = dest_y[i] - this->y[i];
	float len = sqrt(diffX * diffX + diffY * diffY);

	// Making the comparison to see if we need to get the next waypoint
	if (len < this->dest_r[i]) {
		this->dest_x[i] = this->waypoint_x[i][this->waypoint_ptr[i]];
		this->dest_y[i] = this->waypoint_y[i][this->waypoint_ptr[i]];
		this->dest_r[i] = this->waypoint_r[i][this->waypoint_ptr[i]];

		this->waypoint_ptr[i] += 1;
		if (this->waypoint_ptr[i] == this->waypoint_len[i])
			this->waypoint_ptr[i] = 0;
	}
}