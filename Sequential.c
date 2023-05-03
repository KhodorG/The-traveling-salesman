#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_CITIES 10000

int main() {
    int i, j, n, total_cost = 0, cost;
    int **cities;  // Declare cities as a pointer to pointer to int
    int visited[MAX_CITIES] = {0};

    printf("Enter the number of cities (max %d): ", MAX_CITIES);
    scanf("%d", &n);

    if (n > MAX_CITIES) {
        printf("Error: number of cities exceeds maximum allowed.\n");
        exit(1);
    }

    // Allocate memory for the cities array
    cities = (int **)malloc(n * sizeof(int *));
    for (i = 0; i < n; i++) {
        cities[i] = (int *)malloc(n * sizeof(int));
    }

    srand((unsigned)time(NULL));

    printf("Randomly generating distances between cities...\n");
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if (i == j) {
                cities[i][j] = 0;
            } else {
                cities[i][j] = cities[j][i] = rand() % 1000;
            }
        }
    }

    clock_t start_time = clock();

    int min_cost = 99999999;

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

    total_cost = min_cost;

    clock_t end_time = clock();

    // Free the memory allocated for the cities array
    for (i = 0; i < n; i++) {
        free(cities[i]);
    }
    free(cities);

    printf("The minimum cost is %d\n", total_cost);
    printf("Time taken: %f seconds\n", ((double)(end_time - start_time)) / CLOCKS_PER_SEC);

    return 0;
}
