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
#ifndef _ped_model_h_
#define _ped_model_h_ 1

#include <vector>
#include <map>
#include <set>
#include <omp.h>

#include "ped_agent.h"
#include "ped_agents.h"
#include "ped_cuda.h"
#include "ped_region.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ, SEQ1, MOVE_SEQ, MOVE_CONSTANT, MOVE_ADAPTIVE };

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation, int num_threads, double split_factor, double merge_factor);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		std::deque<Region> getRegions() const { return this->regions; };

		// Returns the agents of this scenario
		const std::vector<Tagent*> getAgents() const { return agents; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

	private:

		// The thread tick function for C++ Threads, which moves Tagent between ID low and high
		static void thread_tick(Ped::Model* model, int thread_id);
	
		// The tick function for the CUDA implementation
		static void cuda_tick(Ped::Model* model);

		// The agents in this scenario
		std::vector<Tagent*> agents;
	
		// The destinations in this scenario
		std::vector<Twaypoint*> destinations;

		// The agents array for the general refactored code
		Tagents* agents_array;
		
		// The agents array for the CUDA implementation
		Cuagents cuda_array;

		std::deque<Region> regions;
		// vector<Region> regions;

		void moveAgentsInRegion(int i);
		void moveAgentsInRegionAdaptive(int i);
		std::pair<Region, Region> splitRegion(Region r);
		Ped::Region mergeRegions(Region r1, Region r2);

		double split_factor;
		double merge_factor;

		// Index of agent to move between regions
		deque<int> agent_queue;

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The number of threads used
		int num_threads;
	
		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		// Moves an agent towards its next position
		bool moveLock(Ped::Tagent *agent, int i);

		// Moves an agent towards its next position
		bool moveCAS(Ped::Tagent *agent, int i);

		void addAgentToRegion(int i);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapSeq();
	};
}
#endif
