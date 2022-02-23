//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_cuagents_h_
#define _ped_cuagents_h_ 1

#include "ped_agents.h"
#include "ped_agent.h"

using namespace std;

namespace Ped {
	class Twaypoint;
    class Tagent;
    class Tagents;

	class Cuagents {
	public:
        Cuagents() {};
		Cuagents(Ped::Tagents* agents_array);
		
		// Computing 
        void computeNextDesiredPosition(int i);

		// Freeing the allocated memory
        void free(Ped::Tagents* agents_array);

		// The agents' current positions
		float* x;
		float* y;

		// The agents' desired positions
		float* desiredX;
		float* desiredY;

		// The agents' destinations and the acceptance radius
		float* dest_x;
		float* dest_y;
		float* dest_r;
		
		// The waypoint coordindates, its pointers, lengths, and offsets
		float* waypoint_x;
		float* waypoint_y;
		float* waypoint_r;
		int* waypoint_ptr;
		int* waypoint_len;
		int* waypoint_offset;
	};
}

#endif