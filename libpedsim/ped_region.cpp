#include "ped_region.h"
#include <algorithm>

// Creating a new region with a set upper and lower bound
Ped::Region::Region(int lower_bound, int upper_bound) {
    this->lower_bound = lower_bound;
    this->upper_bound = upper_bound;
}

// Adding an agent to a region
void Ped::Region::addAgent(int i) {
    this->agent_index.push_back(i);
}

// removing an agent
void Ped::Region::removeAgent(int i) {
        std::vector<int>::iterator it;
        // Searching for the agent with index i in the agents_array
        ((it = std::find(this->agent_index.begin(), this->agent_index.end(), i)) == this->agent_index.end()) ? it : this->agent_index.erase(it);
}
