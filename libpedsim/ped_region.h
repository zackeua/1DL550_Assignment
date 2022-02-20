//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_region_h_
#define _ped_region_h_

#include <vector>

namespace Ped {

	class Region
	{
	public:
        // The empty and standard constructors
        Region() {};
        Region(int lower_bound, int upper_bound);
		
        // Adding and removing agents
        void addAgent(int i);
        void removeAgent(int i);

        // Getting the agents from a region
        std::vector<int> getAgents() {return agent_index;};

        // Getting the upper and lower bounds
        int getLowerBound() {return lower_bound;};
        int getUpperBound() {return upper_bound;};

        // Setting the upper and lower bounds
        void setLowerBound(int lower_bound) {this->lower_bound = lower_bound;};
        void setUpperBound(int upper_bound) {this->upper_bound = upper_bound;};

	private:
        int lower_bound;
        int upper_bound;
        std::vector<int> agent_index;
	};
}
#endif
