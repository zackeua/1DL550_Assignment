__global__ void fadeHeatmapCUDA(int* heatmap);
__global__ void incrementHeatCUDA(int numberOfAgents, int* heatmap, float* desiredX, float* desiredY);
