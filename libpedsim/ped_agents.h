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

#ifndef _ped_agents_h_
#define _ped_agents_h_ 1

#include <vector>
#include <deque>

using namespace std;

namespace Ped {
	class Twaypoint;
    class Tagent;

	class Tagents {
	public:
		// The constructor
		Tagents(std::vector<Ped::Tagent*> agents);
	
		// The function computing and setting the new destination and location
        void computeNextDesiredPosition(int i);

		// Same as computeNextDesiredPosition, but with the collision-avoidance part included
        void computeNextDesiredPositionMove(int i);

		// Checks whether or not an agent has reached its destination
		void reachedDestination(int i);

		// The vector of the agents
		std::vector<Ped::Tagent*> agents;

		// The agent's current position
		float* x;
		float* y;
		float* desiredX;
		float* desiredY;
		
		// The agent's destination
		float* dest_x;
		float* dest_y;
		float* dest_r;
		
		// The new coordinatewise waypoint arrays in order to avoid using the deque
		float** waypoint_x;
		float** waypoint_y;
		float** waypoint_r;
		int* waypoint_ptr;
		int* waypoint_len;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*>** waypoints;
	private:
		Tagents() {};  
	};
}

#endif