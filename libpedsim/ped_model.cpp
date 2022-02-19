//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_cuda.h"
#include "ped_model.h"
#include "ped_region.h"

#include <iostream>
#include <stdlib.h>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <math.h>

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>


// The C++ Threads tick function
void Ped::Model::thread_tick(Ped::Model* model, int thread_id) {
	int block_size = model->agents.size() / (model->num_threads);
	int low = thread_id * block_size;
	int high = low + block_size;

	// Giving the remainder to the last thread
	if (thread_id == model->num_threads - 1)
		high = model->agents.size();

	// Looping from low to high within the thread
	for (int i = low; i < high; i++)
		model->agents_array->computeNextDesiredPosition(i);
}

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation, int num_threads)
{
	// Testing if CUDA works on this machine
	cuda_test();

	// Setting up the agents in this scenario
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Setting up the destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Making sure that the number of agents is divisible by four
	if (implementation == IMPLEMENTATION::VECTOR) {
		while (agents.size() % 4 != 0) {
			agents.push_back(agents[0]);
			destinations.push_back(destinations[0]);
		}
	}
	// Allocating the agents array for the general refactored code
	this->agents_array = new Tagents(agents);
	
	// Allocating the agents array for the CUDA implementation
	this->cuda_array = Cuagents(agents_array);		

	// Setting up the chosen implementation. The standard in the given code is SEQ
	this->implementation = implementation;

	// Setting the number of threads
	this->num_threads = num_threads;

	std::cout << &this->regions << std::endl;
	for (int i = 0; i < this->num_threads; i++) {
		Region r = Region(0, 0);
		r.setLowerBound(160/this->num_threads * i);
		r.setUpperBound(160/this->num_threads * (i+1));
		std::cout << r.getLowerBound() << " " << r.getUpperBound() << std::endl;
		this->regions.push_back(r);
	}
	this->regions[this->num_threads-1].setUpperBound(160);

	for (int i = 0; i < agents.size(); i++) {
		for (int j = 0; j < this->num_threads; j++) {
			//std::cout << r.getLowerBound() << " " << r.getUpperBound() << std::endl;

			if (this->regions[j].getLowerBound() <= agents[i]->getX() && agents[i]->getX() < this->regions[j].getUpperBound()) {
				this->regions[j].addAgent(i);
			}
		}
	}
	std::cout << this->agents.size() << std::endl;
	for (int i = 0; i < this->num_threads; i++) {
		std::cout << this->regions[i].getAgents().size() << std::endl;
	}
	/*
	int i;
	std::cin >> i;
	*/
	// The lock variables
	omp_init_lock(&lock);

	// Setting up the heatmap (Relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::tick()
{
	// Choosing the implementation
	switch (this->implementation) {
		case IMPLEMENTATION::SEQ1: // The initial sequential implementation
			for (int i = 0; i < agents.size(); i++) {
				agents[i]->computeNextDesiredPosition();
				agents[i]->setX(agents[i]->getDesiredX());
				agents[i]->setY(agents[i]->getDesiredY());
				
			}
			break;
			
		case IMPLEMENTATION::SEQ: // The refactored sequential implementation
			for (int i = 0; i < agents.size(); i++)
				agents_array->computeNextDesiredPosition(i);
				
			break;

		case IMPLEMENTATION::OMP: // The OpenMP implementation
			// Setting the number of threads as specified earlier
			omp_set_num_threads(this->num_threads);

			// Parallelizing the loop using static scheduling.
			#pragma omp parallel for schedule(static) 
			for (int i = 0; i < agents.size(); i++)
				agents_array->computeNextDesiredPosition(i);
				
			break;
			
		case IMPLEMENTATION::PTHREAD: // The C++ Threads implementation
			{
			// Allocating the threads
			thread* worker = new thread[this->num_threads];
			
			// Running the threads
			for (int i = 0; i < this->num_threads; i++)
				worker[i] = thread(thread_tick, this, i);
			
			// Collecting the results
			for (int i = 0; i < this->num_threads; i++)
				worker[i].join();

			// Deleting the thread list
			delete[] worker;
			}
			break;
		
		case IMPLEMENTATION::VECTOR: // The SIMD implementation
			{
			// Allocating the intrinsics for the vectorization
			__m128 diffX, diffY, sqrt_arg, len, newX, newY, mask1;
			__m128i waypoint_ptr, mask2;

			for (int i = 0; i < agents.size(); i += 4) {
				// Computing the lengths to the destinations
				diffX = _mm_sub_ps(_mm_load_ps(this->agents_array->dest_x + i), _mm_load_ps(this->agents_array->x + i));
				diffY = _mm_sub_ps(_mm_load_ps(this->agents_array->dest_y + i), _mm_load_ps(this->agents_array->y + i));
				sqrt_arg = _mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY));
				len = _mm_sqrt_ps(sqrt_arg);
				
				// Calculating the new x and y positions, and storing them in the x and y arrays
				newX = _mm_add_ps(_mm_load_ps(this->agents_array->x + i), _mm_div_ps(diffX, len));
				newY = _mm_add_ps(_mm_load_ps(this->agents_array->y + i), _mm_div_ps(diffY, len));
				_mm_store_ps(this->agents_array->x + i, _mm_round_ps (newX, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
				_mm_store_ps(this->agents_array->y + i, _mm_round_ps (newY, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));

				// Setting the new x and y coordinates in the graphics component
				this->agents[i]->setX((int)round(this->agents_array->x[i]));
				this->agents[i]->setY((int)round(this->agents_array->y[i]));
				
				this->agents[i+1]->setX((int)round(this->agents_array->x[i+1]));
				this->agents[i+1]->setY((int)round(this->agents_array->y[i+1]));

				this->agents[i+2]->setX((int)round(this->agents_array->x[i+2]));
				this->agents[i+2]->setY((int)round(this->agents_array->y[i+2]));
				
				this->agents[i+3]->setX((int)round(this->agents_array->x[i+3]));
				this->agents[i+3]->setY((int)round(this->agents_array->y[i+3]));

				// Compute the new lengths to the destinations
				diffX = _mm_sub_ps(_mm_load_ps(this->agents_array->dest_x + i), _mm_load_ps(this->agents_array->x + i));
				diffY = _mm_sub_ps(_mm_load_ps(this->agents_array->dest_y + i), _mm_load_ps(this->agents_array->y + i));
				sqrt_arg = _mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY));
				len = _mm_sqrt_ps(sqrt_arg);

				// Determining if each agent has reached its destination, and if so, updating its destination and the waypoint pointer
				if (len[0] < this->agents_array->dest_r[i]) {
					this->agents_array->dest_x[i] = this->agents_array->waypoint_x[i][this->agents_array->waypoint_ptr[i]];
					this->agents_array->dest_y[i] = this->agents_array->waypoint_y[i][this->agents_array->waypoint_ptr[i]];
					this->agents_array->dest_r[i] = this->agents_array->waypoint_r[i][this->agents_array->waypoint_ptr[i]];

					this->agents_array->waypoint_ptr[i] += 1;
					if (this->agents_array->waypoint_ptr[i] == this->agents_array->waypoint_len[i])
						this->agents_array->waypoint_ptr[i] = 0;
				}
				if (len[1] < this->agents_array->dest_r[i+1]) {
					this->agents_array->dest_x[i+1] = this->agents_array->waypoint_x[i+1][this->agents_array->waypoint_ptr[i+1]];
					this->agents_array->dest_y[i+1] = this->agents_array->waypoint_y[i+1][this->agents_array->waypoint_ptr[i+1]];
					this->agents_array->dest_r[i+1] = this->agents_array->waypoint_r[i+1][this->agents_array->waypoint_ptr[i+1]];

					this->agents_array->waypoint_ptr[i+1] += 1;
					if (this->agents_array->waypoint_ptr[i+1] == this->agents_array->waypoint_len[i+1])
						this->agents_array->waypoint_ptr[i+1] = 0;
				}
				if (len[2] < this->agents_array->dest_r[i+2]) {
					this->agents_array->dest_x[i+2] = this->agents_array->waypoint_x[i+2][this->agents_array->waypoint_ptr[i+2]];
					this->agents_array->dest_y[i+2] = this->agents_array->waypoint_y[i+2][this->agents_array->waypoint_ptr[i+2]];
					this->agents_array->dest_r[i+2] = this->agents_array->waypoint_r[i+2][this->agents_array->waypoint_ptr[i+2]];

					this->agents_array->waypoint_ptr[i+2] += 1;
					if (this->agents_array->waypoint_ptr[i+2] == this->agents_array->waypoint_len[i+2])
						this->agents_array->waypoint_ptr[i+2] = 0;
				}
				if (len[3] < this->agents_array->dest_r[i+3]) {
					this->agents_array->dest_x[i+3] = this->agents_array->waypoint_x[i+3][this->agents_array->waypoint_ptr[i+3]];
					this->agents_array->dest_y[i+3] = this->agents_array->waypoint_y[i+3][this->agents_array->waypoint_ptr[i+3]];
					this->agents_array->dest_r[i+3] = this->agents_array->waypoint_r[i+3][this->agents_array->waypoint_ptr[i+3]];

					this->agents_array->waypoint_ptr[i+3] += 1;
					if (this->agents_array->waypoint_ptr[i+3] == this->agents_array->waypoint_len[i+3])
						this->agents_array->waypoint_ptr[i+3] = 0;
				}
			}
		}
		break;

		case IMPLEMENTATION::CUDA: // The CUDA implementation
		{
			cuda_tick(this);
		}
		break;
		
		case IMPLEMENTATION::MOVE_SEQ: // The initial sequential implementation with collision handling
			{
				int k;
			  std::cin >> k;
			for (int i = 0; i < agents.size(); i++) { // This implementation is done
				agents[i]->computeNextDesiredPosition();
				move(agents[i]);
			}

			}
			break;

		case IMPLEMENTATION::MOVE_CAS: // The aligned OpenMP sequential implementation with collision handling
			omp_set_num_threads(this->num_threads);

			#pragma omp parallel
			#pragma omp single
			{
			for (int i = 0; i < this->regions.size(); i++) {
				#pragma omp task
				{
				this->moveAgentsInRegionCAS(i);
				}
			}
			}
			
			while (!this->agent_queue.empty())
			{
				int agent_index = this->agent_queue.front();
				this->agent_queue.pop_front();
				
				agents_array->computeNextDesiredPositionMove(agent_index); 
				move(agents[agent_index]);
				agents_array->reachedDestination(agent_index);
				
				// add the agent back to a region
				this->addAgentToRegion(agent_index);

			}

			// split and merge adjacent regions
			//{			
			// split all regions over some upper threshold

			// merge adjacent regions if they by themself are below lower threshold but dont exceed upper threshold together
			//}

			// print position if two agents are there
			for (int i = 0; i < agents.size(); i++) {
				for (int j = 0; j < i; j++)
				{
					if (agents[i]->getX() == agents[j]->getX() && agents[i]->getY() == agents[j]->getY())
						std::cout << agents[i]->getX() << " " << agents[i]->getY() << std::endl;
				}
				
			}

			break;

		case IMPLEMENTATION::MOVE_LOCK: // The aligned sequential implementation  with collision handling
			 {
			  int k;
			  std::cin >> k;
			omp_set_num_threads(this->num_threads);
			// Parallelizing the loop using static scheduling.
			#pragma omp parallel
			{

			#pragma omp single
			{	
				for (int i = 0; i < regions.size(); i++) {

				#pragma omp task
				{
					this->moveAgentsInRegion(i);
				}
				}
			}
			}

			for (int i = 0; i < agents.size(); i++) {
				for (int j = 0; j < i; j++)
				{
					if (agents[i]->getX() == agents[j]->getX() && agents[i]->getY() == agents[j]->getY())
						std::cout << agents[i]->getX() << " " << agents[i]->getY() << std::endl;
				}
				
			}

			

			while (!this->agent_queue.empty())
			{
				int agent_index = this->agent_queue.front();
				this->agent_queue.pop_front();
				
				agents_array->computeNextDesiredPositionMove(agent_index);
				move(agents[agent_index]);
				agents_array->reachedDestination(agent_index);
				
				// add the agent back to a region
				if (agents_array->x[agent_index] <= 40) {
					this->regions[0].addAgent(agent_index);
				} else if (agents_array->x[agent_index] <= 80) {
					this->regions[1].addAgent(agent_index);
				} else if (agents_array->x[agent_index] <= 120) {
					this->regions[2].addAgent(agent_index);
				} else {
					this->regions[3].addAgent(agent_index);
				}
			}
			
			 }
			break;
	}
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void Ped::Model::moveAgentsInRegion(int i) {
	//std::cout << this->regions[i].getAgents().size() << std::endl;

	for (int j = 0; j < this->regions[i].getAgents().size(); j++) {
		int agent_id = this->regions[i].getAgents()[j];
		agents_array->computeNextDesiredPositionMove(agent_id);
		if (moveLock(agents[agent_id], agent_id)) {
			agents_array->reachedDestination(agent_id);
		} else {
			this->regions[i].removeAgent(agent_id);
		}
	}
}


void Ped::Model::moveAgentsInRegionCAS(int i) {
	//std::cout << this->regions[i].getAgents().size() << std::endl;

	for (int j = 0; j < this->regions.at(i).getAgents().size(); j++) {
		int agent_id = this->regions.at(i).getAgents()[j];
		agents_array->computeNextDesiredPositionMove(agent_id);
		if (moveCAS(agents[agent_id], agent_id)) {
			agents_array->reachedDestination(agent_id);
		} else {
			this->regions.at(i).removeAgent(agent_id);
		}
	}
}



void Ped::Model::addAgentToRegion(int i) {
	for (int j = 0; j < this->regions.size(); j++) {
		if (this->agents_array->x[i] <= this->regions.at(j).getUpperBound()) {
			this->regions[j].addAgent(i);
			return;
		}
	}

	// just in case the if statement in the loop do not work
	// requires the regions to be sorted from lower x to higher x values
	//this->regions[this->regions.size()-1].addAgent(i);
}


// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
bool Ped::Model::moveLock(Ped::Tagent *agent, int i)
{
	//? Should we keep everything as agent objects?
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Questions:
	// 1. How do we make each task know which agents it owns?
	// 2. How do we make each task know which region it governs?
	// 3. How do we update the size and the number of regions automatically?

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	int off = 3;
		for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {
			
			// TODO: Lock this while parallelizing this
			// If the current position is not yet taken by any neighbor
			if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {				
				// Set the agent's position
				// change to a list afterwards instead of omp critical
				if (		0 <= agent->getX() && agent->getX() < 40 && 40-off <= (*it).first || 40 <= agent->getX() && agent->getX() < 80 && (*it).first < 40+off) {
					this->agent_queue.push_back(i);
					return false;
				} else if (40 <= agent->getX() && agent->getX() < 80 && 80-off <= (*it).first || 80 <= agent->getX() && agent->getX() < 120 && (*it).first < 80+off) {	
					this->agent_queue.push_back(i);
					return false;
				} else if (80 <= agent->getX() && agent->getX() < 120 && 120-off <= (*it).first || 120 <= agent->getX() && agent->getX() < 160 && (*it).first < 120+off) {
					this->agent_queue.push_back(i);
					return false;
				} else {
					agent->setX((*it).first);
					agent->setY((*it).second);
					return true;
				}
			}
		}
	return true;
}


// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
bool Ped::Model::moveCAS(Ped::Tagent *agent, int i)
{
	//? Should we keep everything as agent objects?
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	int off = 3;
	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {
			for (int j = 0; j < this->regions.size(); j++) {
				if (j == 0 && agent->getX() < this->regions.at(j).getUpperBound() && this->regions.at(j).getUpperBound() - off <= (*it).first) {
					this->agent_queue.push_back(i);
					return false;
				} else if (j == this->regions.size()-1 && this->regions.at(j).getLowerBound() <= agent->getX() && (*it).first <= this->regions.at(j).getLowerBound() + off) {
					this->agent_queue.push_back(i);
					return false;
				} else if (this->regions.at(j).getLowerBound() <= agent->getX() && agent->getX() < this->regions.at(j).getUpperBound() && (this->regions.at(j).getUpperBound() - off <= (*it).first || (*it).first <= this->regions.at(j).getLowerBound() + off)) {
					this->agent_queue.push_back(i);
					return false;
				}
			}

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);
			return true;
			
		}
	}
	return true;

}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	//? Should we keep everything as agent objects?
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {
	// TODO: Don't include all agents as your neighbors.
	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	this->cuda_array.free(agents_array);
	//std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	//std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
