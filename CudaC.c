#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_CITIES 5000
#define THREADS_PER_BLOCK 512
#defin MAX_DISTANCE 100

__global__ void calculate_minimum_cost(int *d_cities, int n, int *d_min_cost)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int visited[MAX_CITIES] = {0};
    int cost, min_cost = 99999999;
    __shared__ int s_cities[MAX_CITIES];

    // Copy cities to shared memory
    for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
       int idx = i + threadIdx.x;
       if (idx < n) {
          s_cities[idx] = d_cities[tid * n + idx];
        }
    }
    __syncthreads();

    if (tid < n) {
        visited[tid] = 1;
        cost = 0;

        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                cost += s_cities[j];
            }
        }

        if (cost < min_cost) {
            min_cost = cost;
        }

        visited[tid] = 0;
    }

    atomicMin(d_min_cost, min_cost);
}

int main()
{
    int n = MAX_CITIES;
    int *cities;
    int min_cost = 99999999, total_cost;
    int *d_cities, *d_min_cost;

    // Allocate memory for cities
    cities = (int *)malloc(n * n * sizeof(int));

    // Initialize distances between cities
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            if (i == j) {
                cities[i * n + j] = 0;
            } else {
                cities[i * n + j] = cities[j * n + i] = MAX_DISTANCE; // Fixed distance of 100 between cities
            }
        }
    }

    // Copy cities to device
    cudaMalloc((void **)&d_cities, n * n * sizeof(int));
    cudaMemcpy(d_cities, cities, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory for minimum cost on device
    cudaMalloc((void **)&d_min_cost, sizeof(int));
    cudaMemcpy(d_min_cost, &min_cost, sizeof(int), cudaMemcpyHostToDevice);

    // Get start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel
    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    calculate_minimum_cost<<<num_blocks, THREADS_PER_BLOCK>>>(d_cities, n, d_min_cost);

    // Get end time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Copy minimum cost from device to host
    cudaMemcpy(&min_cost, d_min_cost, sizeof(int), cudaMemcpyDeviceToHost);

    total_cost = min_cost;

    printf("The minimum cost is %d\n", total_cost);
    printf("Time taken: %f seconds\n", elapsed_time / 1000.0);

    // Free memory
    free(cities);
    cudaFree(d_cities);
    cudaFree(d_min_cost);

    return 0;
}
