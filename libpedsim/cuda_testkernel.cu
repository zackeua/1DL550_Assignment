#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "ped_agents.h"
#include "ped_waypoint.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}


__global__ void print_func() {
	printf("Hello world from thread %d\n", 
	blockIdx.x * blockDim.x + threadIdx.x);
}

void hello()
{
	print_func <<<2, 10>>> ();
}

/*
__global__ void cuda_func(int n, Ped::Tagents* agents_array) {
	int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < n; i += stride) {
		//agents_array->computeNextDesiredPosition(i);

		Ped::Twaypoint* nextDestination = NULL;
		bool agentReachedDestination = false;

		if (agents_array->destination[i] != NULL) {
			// compute if agent reached its current destination
			

			double diffX = agents_array->dest_x[i] - agents_array->x[i];
			double diffY = agents_array->dest_y[i] - agents_array->y[i];
			double length = sqrt(diffX * diffX + diffY * diffY);
			agentReachedDestination = length < agents_array->dest_r[i];
			//std::cout << " " << this->x[i] << std::endl;
			//std::cout << " " << this->y[i] << std::endl;
			//std::cout << length << " " << this->destination[i]->getr() << std::endl;
		}

		if ((agentReachedDestination || agents_array->destination[i] == NULL) && !agents_array->waypoints[i]->empty()) {
			// Case 1: agent has reached destination (or has no current destination);
			// get next destination if available
			if (agents_array->destination[i] != NULL) {
				agents_array->waypoints[i]->push_back(agents_array->destination[i]);
			}
			nextDestination = agents_array->waypoints[i]->front();
			agents_array->dest_x[i] = nextDestination->x;
			agents_array->dest_y[i] = nextDestination->y;
			agents_array->dest_r[i] = nextDestination->r;

			agents_array->waypoints[i]->pop_front();
			// DO NOT print destination here, might be NULL
		}
		else {
			// Case 2: agent has not yet reached destination, continue to move towards
			// current destination
			nextDestination = agents_array->destination[i];
		}

		agents_array->destination[i] = nextDestination;


		if (agents_array->destination[i] == NULL) {
			// no destination, no need to
			// compute where to move to
			return;
		}
		// Safe to print here
		//std::cout << this->destination[i]->getx() << std::endl;	
		//std::cout << this->destination[i]->gety() << std::endl;

		//double diffX = destination[i]->getx() - this->x[i];
		//double diffY = destination[i]->gety() - this->y[i];

		// SIMD: recleare diffX and diffY as simd
		double diffX = agents_array->dest_x[i] - agents_array->x[i];
		double diffY = agents_array->dest_y[i] - agents_array->y[i];

		double len = sqrt(diffX * diffX + diffY * diffY);
		
		// SIMD:
		agents_array->x[i] = (int)round(agents_array->x[i] + diffX / len);
		agents_array->y[i] = (int)round(agents_array->y[i] + diffY / len);

		//agents_array->agents[i]->setX(agents_array->x[i]);
		//agents_array->agents[i]->setY(agents_array->y[i]);

	}
}


int cuda_tick(Ped::Tagents* agents) {
	Ped::Tagents* cuda_agents;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	cudaStatus = cudaMallocManaged((void**)&cuda_agents, sizeof(agents));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!\n");
		return 1;
	}

	int number_of_blocks = 1;
	int threads_per_block = 1;
	cuda_func <<<number_of_blocks, threads_per_block>>> (cuda_agents->agents.size(), cuda_agents);

	cudaStatus = cudaDeviceSynchronize();

	cudaFree(cuda_agents);

	return 0;

}
*/

int cuda_test()
{
    static int tested = 0;

	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

    if (tested == 1)
        return 0;
    tested = 1;

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!\n");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else
	{
		//fprintf(stderr, "Cuda launch succeeded! \n");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	if (cudaStatus != 0){
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	}
	else{
		fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing
	}

	return cudaStatus;
}
