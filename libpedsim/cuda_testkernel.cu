#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "ped_agents.h"
#include "ped_agent.h"
#include "ped_cuda.h"
#include "ped_model.h"

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

__global__ void cuda_func(int n, Ped::Cuagents* cuda_agents) {
	
	//printf("Hello ogge from thread %d\n", 
	//blockIdx.x * blockDim.x + threadIdx.x);

	//printf("Index: %d, n = %d\n", blockIdx.x * blockDim.x + threadIdx.x, n);
	float f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (blockIdx.x+1) * blockDim.x) {
		double diffX = cuda_agents->dest_x[i] - cuda_agents->x[i];
		double diffY = cuda_agents->dest_y[i] - cuda_agents->y[i];

		double len = sqrt(diffX * diffX + diffY * diffY);

		cuda_agents->x[i] = (int)round(cuda_agents->x[i] + diffX / len);
		cuda_agents->y[i] = (int)round(cuda_agents->y[i] + diffY / len);

		//printf("%f\n", cuda_agents->x[i]);

		// If the destination is null, or if the agent has reached its destination, then we compute its new destination coordinates.
		if (len < cuda_agents->dest_r[i]) {
			cuda_agents->dest_x[i] = cuda_agents->waypoint_x[i][cuda_agents->waypoint_ptr[i]];
			cuda_agents->dest_y[i] = cuda_agents->waypoint_y[i][cuda_agents->waypoint_ptr[i]];
			cuda_agents->dest_r[i] = cuda_agents->waypoint_r[i][cuda_agents->waypoint_ptr[i]];

			cuda_agents->waypoint_ptr[i] += 1;
			if (cuda_agents->waypoint_ptr[i] == cuda_agents->waypoint_len[i])
				cuda_agents->waypoint_ptr[i] = 0;
		}
	}
}

void Ped::Model::cuda_tick(Ped::Model* model) {



	cudaError_t cudaStatus;



	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!\n");
		return;
	}

	

	int number_of_blocks = 2;
	int threads_per_block = 100;
	cuda_func <<<number_of_blocks, threads_per_block>>> (model->agents.size(), &model->cuda_array);
	
	cudaStatus = cudaDeviceSynchronize();
	
	cudaStatus = cudaMemcpy((void**)&model->agents_array->x, (void**)&model->cuda_array.x, model->agents.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%d\n", cudaStatus);
		fprintf(stderr, "addWithCuda1 failed!\n");
		return;
	}
	cudaStatus = cudaMemcpy(model->agents_array->y, model->cuda_array.y, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda2 failed!\n");
		return;
	}
	//printf("Cuda: (%f, %f)\n", model->cuda_array.x[0], model->cuda_array.y[0]);

	printf("Model: (%f, %f)\n", model->agents_array->x[0], model->agents_array->y[0]);

	//printf("Agent: (%f, %f)\n", model->agents[0]->getX(), model->agents[0]->getY());

	for (int i = 0; i < model->agents.size(); i++) {
		
		//model->agents[i]->setX(0);
		//model->agents[i]->setY(0);

		model->agents[i]->setX(model->agents_array->x[i]);
		model->agents[i]->setY(model->agents_array->y[i]);
	}
	//printf("agents->agents.size(): %d\n", model->agents.size());

}

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
