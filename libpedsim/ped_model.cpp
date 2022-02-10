//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>

#include <math.h>

#include <xmmintrin.h>
#include <smmintrin.h>

#include <stdlib.h>

void Ped::Model::thread_tick(Ped::Model* model, int thread_id) {
	int block_size = model->agents.size()/(model->num_threads);
	int low = thread_id * block_size;
	int high = low + block_size;

	if (thread_id == model->num_threads-1)
		high = model->agents.size();

	for (int i = low; i < high; i++)
		model->agents_array->computeNextDesiredPosition(i);
}

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation, int num_threads)
{
	// Convenience test: does CUDA work on this machine?
	//cuda_test();

	// Set
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	if (implementation == IMPLEMENTATION::VECTOR) {
		while (agents.size() % 4 != 0) {
			agents.push_back(agents[0]);
			destinations.push_back(destinations[0]);
		}
	}

	// Initializing the array of agents
	this->agents_array = new Tagents(agents);

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Sets the number of threads
	this->num_threads = num_threads;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::tick()
{
	// Toggling which case to run
	switch (this->implementation) {
		case IMPLEMENTATION::SEQ: // The sequential version
			for (int i = 0; i < agents.size(); i++)
				agents_array->computeNextDesiredPosition(i);
			break;

		case IMPLEMENTATION::OMP: // The OpenMP version 
			// Setting the number of threads
			omp_set_num_threads(this->num_threads);

			// Choosing the scheduling technique
			#pragma omp parallel for schedule(static)

			// Looping over the array according to OpenMP.
			for (int i = 0; i < agents.size(); i++)
				agents_array->computeNextDesiredPosition(i);
			break;
			
		case IMPLEMENTATION::PTHREAD: { // The C++ Threads version
			// Creating a pointer to the thread array.
			thread* worker = new thread[this->num_threads];
			
			// Creating the threads and running them
			for (int i = 0; i < this->num_threads; i++)
				worker[i] = thread(thread_tick, this, i);
			
			// Killing the threads
			for (int i = 0; i < this->num_threads; i++)
				worker[i].join();

			// Freeing the thread array.
			delete[] worker;
			}
			break;
		
		case IMPLEMENTATION::VECTOR: // The vectorized version

			for (int i = 0; i < agents.size(); i += 4) {

				//? Reference: https://cs.stackexchange.com/questions/86154/if-and-while-statements-within-a-simd-architecture-and-its-memory-architectu
				/*
				 * TODO: SIMD
				 * 1. Check if you can vectorize TODO-1 and TODO-4
				 * 2. Check if you can vectorize TODO-5
				 * 3. Check if you can remove the if statements for TODO-2 and TODO-3
				 */
				/*
				 * TODO: Friday
				 * 1. Finish implementing and testing SIMD
				 * 2. Finish implementing and testing CUDA
				 * 3. Write the report and submit it
				 */

				////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				// TODO 1: Can't vectorize these lines since they aren't vectors
				Ped::Twaypoint* nextDestination0 = NULL;
				Ped::Twaypoint* nextDestination1 = NULL;
				Ped::Twaypoint* nextDestination2 = NULL;
				Ped::Twaypoint* nextDestination3 = NULL;
				bool agentReachedDestination0 = false;
				bool agentReachedDestination1 = false;
				bool agentReachedDestination2 = false;
				bool agentReachedDestination3 = false;

				// TODO 2: Can't vectorize these lines because they are if statements
				if (this->agents_array->destination[i] != NULL) {
					// Compute if agent reached its current destination
					double diffX0 = this->agents_array->dest_x[i] - this->agents_array->x[i];
					double diffY0 = this->agents_array->dest_y[i] - this->agents_array->y[i];
					double length0 = sqrt(diffX0 * diffX0 + diffY0 * diffY0);
					agentReachedDestination0 = length0 < this->agents_array->dest_r[i];
				}
				if (this->agents_array->destination[i+1] != NULL) {
					// Compute if agent reached its current destination
					double diffX1 = this->agents_array->dest_x[i+1] - this->agents_array->x[i+1];
					double diffY1 = this->agents_array->dest_y[i+1] - this->agents_array->y[i+1];
					double length1 = sqrt(diffX1 * diffX1 + diffY1 * diffY1);
					agentReachedDestination1 = length1 < this->agents_array->dest_r[i+1];
				}
				if (this->agents_array->destination[i+2] != NULL) {
					// Compute if agent reached its current destination
					double diffX2 = this->agents_array->dest_x[i+2] - this->agents_array->x[i+2];
					double diffY2 = this->agents_array->dest_y[i+2] - this->agents_array->y[i+2];
					double length2 = sqrt(diffX2 * diffX2 + diffY2 * diffY2);
					agentReachedDestination2 = length2 < this->agents_array->dest_r[i+2];
				}
				if (this->agents_array->destination[i+3] != NULL) {
					// Compute if agent reached its current destination
					double diffX3 = this->agents_array->dest_x[i+3] - this->agents_array->x[i+3];
					double diffY3 = this->agents_array->dest_y[i+3] - this->agents_array->y[i+3];
					double length3 = sqrt(diffX3 * diffX3 + diffY3 * diffY3);
					agentReachedDestination3 = length3 < this->agents_array->dest_r[i+3];
				}

				//////////////////////////////////////////////////////

				// TODO 3: Can't vectorize these lines because they are if statements
				if ((agentReachedDestination0 || this->agents_array->destination[i] == NULL) && !this->agents_array->waypoints[i]->empty()) {
					// Case 1: Agent has reached destination (or has no current destination);
					// get next destination if available
					if (this->agents_array->destination[i] != NULL) {
						this->agents_array->waypoints[i]->push_back(this->agents_array->destination[i]);
					}
					nextDestination0 = this->agents_array->waypoints[i]->front();
					this->agents_array->dest_x[i] = nextDestination0->getx();
					this->agents_array->dest_y[i] = nextDestination0->gety();
					this->agents_array->dest_r[i] = nextDestination0->getr();

					
				// double diffX0 = this->agents_array->dest_x[i] - this->agents_array->x[i];
				// double diffY0 = this->agents_array->dest_y[i] - this->agents_array->y[i];

					this->agents_array->waypoints[i]->pop_front();
					// Do not print this->destination here since it might be NULL
				} //////////////////////////////////////////////////////
				else {
					// Case 2: Agent has not yet reached destination, continue to move towards
					// current destination
					nextDestination0 = this->agents_array->destination[i];
				}

				if ((agentReachedDestination1 || this->agents_array->destination[i+1] == NULL) && !this->agents_array->waypoints[i+1]->empty()) {
					// Case 1: Agent has reached destination (or has no current destination);
					// get next destination if available
					if (this->agents_array->destination[i+1] != NULL) {
						this->agents_array->waypoints[i+1]->push_back(this->agents_array->destination[i+1]);
					}
					nextDestination1 = this->agents_array->waypoints[i+1]->front();
					this->agents_array->dest_x[i+1] = nextDestination1->getx();
					this->agents_array->dest_y[i+1] = nextDestination1->gety();
					this->agents_array->dest_r[i+1] = nextDestination1->getr();

					this->agents_array->waypoints[i+1]->pop_front();
					// Do not print this->destination here since it might be NULL
				} //////////////////////////////////////////////////////
				else {
					// Case 2: Agent has not yet reached destination, continue to move towards
					// current destination
					nextDestination1 = this->agents_array->destination[i+1];
				}

				if ((agentReachedDestination2 || this->agents_array->destination[i+2] == NULL) && !this->agents_array->waypoints[i+2]->empty()) {
					// Case 1: Agent has reached destination (or has no current destination);
					// get next destination if available
					if (this->agents_array->destination[i+2] != NULL) {
						this->agents_array->waypoints[i+2]->push_back(this->agents_array->destination[i+2]);
					}
					nextDestination2 = this->agents_array->waypoints[i+2]->front();
					this->agents_array->dest_x[i+2] = nextDestination2->getx();
					this->agents_array->dest_y[i+2] = nextDestination2->gety();
					this->agents_array->dest_r[i+2] = nextDestination2->getr();

					this->agents_array->waypoints[i+2]->pop_front();
					// Do not print this->destination here since it might be NULL
				} //////////////////////////////////////////////////////
				else {
					// Case 2: Agent has not yet reached destination, continue to move towards
					// current destination
					nextDestination2 = this->agents_array->destination[i+2];
				}

				if ((agentReachedDestination3 || this->agents_array->destination[i+3] == NULL) && !this->agents_array->waypoints[i+3]->empty()) {
					// Case 1: Agent has reached destination (or has no current destination);
					// get next destination if available
					if (this->agents_array->destination[i+3] != NULL) {
						this->agents_array->waypoints[i+3]->push_back(this->agents_array->destination[i+3]);
					}
					nextDestination3 = this->agents_array->waypoints[i+3]->front();
					this->agents_array->dest_x[i+3] = nextDestination3->getx();
					this->agents_array->dest_y[i+3] = nextDestination3->gety();
					this->agents_array->dest_r[i+3] = nextDestination3->getr();

					this->agents_array->waypoints[i+3]->pop_front();
					// Do not print this->destination here since it might be NULL
				} //////////////////////////////////////////////////////
				else {
					// Case 2: Agent has not yet reached destination, continue to move towards
					// current destination
					nextDestination3 = this->agents_array->destination[i+3];
				}

				////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				// TODO 4: Can't vectorize these lines since they aren't vectors
				// Setting the agent destinations
				this->agents_array->destination[i] = nextDestination0;
				this->agents_array->destination[i+1] = nextDestination1;
				this->agents_array->destination[i+2] = nextDestination2;
				this->agents_array->destination[i+3] = nextDestination3;

				// If it is null, then we abort that update
				if (this->agents_array->destination[i] == NULL) {return;}
				if (this->agents_array->destination[i+1] == NULL) {return;}
				if (this->agents_array->destination[i+2] == NULL) {return;}
				if (this->agents_array->destination[i+3] == NULL) {return;}

				// The SISD version of the below
				/*
				double diffX0 = this->agents_array->dest_x[i] - this->agents_array->x[i];
				double diffY0 = this->agents_array->dest_y[i] - this->agents_array->y[i];

				double diffX1 = this->agents_array->dest_x[i+1] - this->agents_array->x[i+1];
				double diffY1 = this->agents_array->dest_y[i+1] - this->agents_array->y[i+1];

				double diffX2 = this->agents_array->dest_x[i+2] - this->agents_array->x[i+2];
				double diffY2 = this->agents_array->dest_y[i+2] - this->agents_array->y[i+2];

				double diffX3 = this->agents_array->dest_x[i+3] - this->agents_array->x[i+3];
				double diffY3 = this->agents_array->dest_y[i+3] - this->agents_array->y[i+3];
				*/

				// Computing the difference in each coordinate
				__m128 diffX = _mm_sub_ps(_mm_load_ps(this->agents_array->dest_x + i), _mm_load_ps(this->agents_array->x + i));
				__m128 diffY = _mm_sub_ps(_mm_load_ps(this->agents_array->dest_y + i), _mm_load_ps(this->agents_array->y + i));
				
				// The SISD version of the below
				/*
				double len0 = sqrt(diffX0 * diffX0 + diffY0 * diffY0);
				double len1 = sqrt(diffX1 * diffX1 + diffY1 * diffY1);
				double len2 = sqrt(diffX2 * diffX2 + diffY2 * diffY2);
				double len3 = sqrt(diffX3 * diffX3 + diffY3 * diffY3);
				*/

				// Compute the length using the square root
				__m128 sqrt_arg = _mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY));
				__m128 len = _mm_mul_ps(sqrt_arg, _mm_rsqrt_ps(sqrt_arg));
				//sqrt_arg * 1/sqrt(sqrt_arg) <- faster
				//__m128 len = _mm_sqrt_ps(sqrt_arg);

				// The SISD version of the below
				/*
				this->agents_array->x[i] = (int)round(this->agents_array->x[i] + diffX0 / len0);
				this->agents_array->y[i] = round(this->agents_array->y[i] + diffY0 / len0);

				this->agents_array->x[i+1] = round(this->agents_array->x[i+1] + diffX1 / len1);
				this->agents_array->y[i+1] = round(this->agents_array->y[i+1] + diffY1 / len1);

				this->agents_array->x[i+2] = round(this->agents_array->x[i+2] + diffX2 / len2);
				this->agents_array->y[i+2] = round(this->agents_array->y[i+2] + diffY2 / len2);

				this->agents_array->x[i+3] = round(this->agents_array->x[i+3] + diffX3 / len3);
				this->agents_array->y[i+3] = round(this->agents_array->y[i+3] + diffY3 / len3);
				*/

				// Store the results
				__m128 newX = _mm_add_ps(_mm_load_ps(this->agents_array->x + i), _mm_div_ps(diffX, len));
				__m128 newY = _mm_add_ps(_mm_load_ps(this->agents_array->y + i), _mm_div_ps(diffY, len));

				// Round off the results
				_mm_store_ps(this->agents_array->x + i, _mm_round_ps(newX, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
				_mm_store_ps(this->agents_array->y + i, _mm_round_ps(newY, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));

				// TODO 5: Can't vectorize these lines since they aren't vectors
				// Comment this in order to not get overhead for updating the graphics (Can't vectorize this because of the function call)
				this->agents[i]->setX((int)round(this->agents_array->x[i]));
				this->agents[i]->setY((int)round(this->agents_array->y[i]));
				
				this->agents[i+1]->setX((int)round(this->agents_array->x[i+1]));
				this->agents[i+1]->setY((int)round(this->agents_array->y[i+1]));

				this->agents[i+2]->setX((int)round(this->agents_array->x[i+2]));
				this->agents[i+2]->setY((int)round(this->agents_array->y[i+2]));
				
				this->agents[i+3]->setX((int)round(this->agents_array->x[i+3]));
				this->agents[i+3]->setY((int)round(this->agents_array->y[i+3]));
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
	// std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	// std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
