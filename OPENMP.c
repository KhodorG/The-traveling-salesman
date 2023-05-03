#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define MAX_CITIES 1000

int main(int argc, char *argv[]) {
    int i, j, n, total_cost = 0, cost;
    int **cities;
    int min_cost = 99999999;

    double start_time = omp_get_wtime();

    printf("Enter the number of cities (max %d): ", MAX_CITIES);
    scanf("%d", &n);

    if (n > MAX_CITIES) {
        printf("Error: number of cities exceeds maximum allowed.\n");
        exit(1);
    }

    printf("Randomly generating distances between cities...\n");

    omp_set_num_threads(4);

    // Allocate cities array on the heap
    cities = (int **)malloc(n * sizeof(int *));
    for (i = 0; i < n; i++) {
        cities[i] = (int *)malloc(n * sizeof(int));
    }

    #pragma omp parallel private(i, j, cost)
    {
        int visited[MAX_CITIES] = {0};
        unsigned int seed = time(NULL) + omp_get_thread_num();

        #pragma omp for collapse(2)
        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                if (i == j) {
                    cities[i][j] = 0;
                } else {
                    cities[i][j] = cities[j][i] = rand_r(&seed) % 100;
                }
            }
        }

        #pragma omp for reduction(min:min_cost)
        for (i = 0; i < n; i++) {
            visited[i] = 1;
            cost = 0;

            for (j = 0; j < n; j++) {
                if (!visited[j]) {
                    cost += cities[i][j];
                }
            }

            if (cost < min_cost) {
                min_cost = cost;
            }

            visited[i] = 0;
        }
    }

    total_cost = min_cost;

    double end_time = omp_get_wtime();

    printf("The minimum cost is %d\n", total_cost);
    printf("Time taken: %f seconds\n", end_time - start_time);

    // Free cities array memory
    for (i = 0; i < n; i++) {
        free(cities[i]);
    }
    free(cities);

    return 0;
}
