__global__ void fadeHeatmapCUDA(int* heatmap);
__global__ void incrementHeatCUDA(int numberOfAgents, int* heatmap, float* desiredX, float* desiredY);
__global__ void capHeatmapCUDA(int* heatmap);
__global__ void scaledHeatmapCUDA(int* heatmap, int* scaled_heatmap);
__global__ void blurredHeatmapCUDA(int* scaled_heatmap, int* blurred_cuda);


