#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "ped_agents.h"
#include "ped_agent.h"
#include "ped_cuda.h"
#include "ped_model.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void cuda_func(int n, float* x, float* y, float* dest_x, float* dest_y, float* dest_r, float* waypoint_x, float* waypoint_y, float* waypoint_r, int* waypoint_ptr, int* waypoint_len, int* waypoint_offset) {

	// Running the simulation, starting with the current thread in the block, and then jumping to that block in the next grid
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		
		// Computing the lengths to the destinations
		double diffX = dest_x[i] - x[i];
		double diffY = dest_y[i] - y[i];
		double len = sqrt(diffX * diffX + diffY * diffY);

		// Calculating the new x and y positions, and storing them in the x and y arrays
		x[i] = (int)round(x[i] + diffX / len);
		y[i] = (int)round(y[i] + diffY / len);

		// Compute the new lengths to the destinations
		diffX = dest_x[i] - x[i];
		diffY = dest_y[i] - y[i];
		len = sqrt(diffX * diffX + diffY * diffY);
		
		// Determining if each agent has reached its destination, and if so, updating its destination and the waypoint pointer
		if (len < dest_r[i]) {
			dest_x[i] = waypoint_x[waypoint_offset[i] + waypoint_ptr[i]];
			dest_y[i] = waypoint_y[waypoint_offset[i] + waypoint_ptr[i]];
			dest_r[i] = waypoint_r[waypoint_offset[i] + waypoint_ptr[i]];

			waypoint_ptr[i] += 1;
			if (waypoint_ptr[i] == waypoint_len[i])
				waypoint_ptr[i] = 0;
		}

	}
}

void Ped::Model::cuda_tick(Ped::Model* model) {
	// Allocating the CUDA status
	cudaError_t cudaStatus;

	// Setting the CUDA device
	cudaStatus = cudaSetDevice(0);

	// Checking if that worked
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\n");
		return;
	}

	// Setting the number of threads
	int number_of_blocks = 100;
	int threads_per_block = 100;
	
	// Running the CUDA implementatin on the GPU
	cuda_func <<<number_of_blocks, threads_per_block>>> (model->agents.size(), model->cuda_array.x, model->cuda_array.y, model->cuda_array.dest_x, model->cuda_array.dest_y, model->cuda_array.dest_r, model->cuda_array.waypoint_x, model->cuda_array.waypoint_y, model->cuda_array.waypoint_r, model->cuda_array.waypoint_ptr, model->cuda_array.waypoint_len, model->cuda_array.waypoint_offset);
	
	// Synchronizing the threads
	cudaStatus = cudaDeviceSynchronize();
	
	// Copying back the results for the x value back to the processor
	cudaStatus = cudaMemcpy(model->agents_array->x, model->cuda_array.x, model->agents.size() * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Checking if that worked
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%d\n", cudaStatus);
		fprintf(stderr, "cudaMemcpy1 failed!\n");
		return;
	}
	
	// Copying back the results for the x value back to the processor
	cudaStatus = cudaMemcpy(model->agents_array->y, model->cuda_array.y, model->agents.size() * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Checking if that worked
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		return;
	}

	// Setting the new x and y coordinates in the graphics component
	for (int i = 0; i < model->agents.size(); i++) {
		model->agents[i]->setX(model->agents_array->x[i]);
		model->agents[i]->setY(model->agents_array->y[i]);
	}
}

// Verifying that CUDA works on this machine
int cuda_test() {
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
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
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
	} else {
		// fprintf(stderr, "Cuda launch succeeded! \n");
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

	if (cudaStatus != 0) {
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	} else {
		fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing
	}

	return cudaStatus;
}
