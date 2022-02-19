
#include "ped_region.h"
#include <algorithm>



Ped::Region::Region(int lower_bound, int upper_bound) {
    this->lower_bound = lower_bound;
    this->upper_bound = upper_bound;
}


void Ped::Region::addAgent(int i) {
    this->agent_index.push_back(i);
}

void Ped::Region::removeAgent(int i) {
        std::vector<int>::iterator it;

        ((it = std::find(this->agent_index.begin(), this->agent_index.end(), i)) == this->agent_index.end()) ? it : this->agent_index.erase(it);

}
