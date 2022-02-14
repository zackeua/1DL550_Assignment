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
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
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
