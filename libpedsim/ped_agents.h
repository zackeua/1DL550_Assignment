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
		Tagents(std::vector<Ped::Tagent*> agents);
		

        void computeNextDesiredPosition(int i);
        
        void addWaypoint(deque<Twaypoint*>* wp, int i);

		std::vector<Ped::Tagent*> agents;

		// The agent's current position
		float* x;
		float* y;


		// The agent's destination
		float* dest_x;
		float* dest_y;
		float* dest_r;

		
		float** waypoint_x;
		float** waypoint_y;
		float** waypoint_r;
		int* waypoint_ptr;
		int* waypoint_len;
		


		// The agent's desired next position
		float* desiredPositionX;
		float* desiredPositionY;

			// The current destination (may require several steps to reach)
		Twaypoint** destination;

		// The last destination
		Twaypoint** lastDestination;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*>** waypoints;

		Ped::Twaypoint* getNextDestination(int i);
	private:
		Tagents() {};

		

	

        
	};
}

#endif