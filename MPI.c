#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define MAX_CITIES 10000
#define MAX_DISTANCE 100

int main(int argc, char *argv[]) {
    int rank, size, i, j, n, total_cost = 0, cost;
    int **cities;
    int visited[MAX_CITIES] = {0};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    if (rank == 0) {
        printf("Enter the number of cities (max %d): ", MAX_CITIES);
        scanf("%d", &n);

        cities = (int **) malloc(n * sizeof(int *));
        for (i = 0; i < n; i++) {
            cities[i] = (int *) malloc(n * sizeof(int));
        }

        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                if (i == j) {
                    cities[i][j] = 0;
                } else {
                    cities[i][j] = cities[j][i] = MAX_DISTANCE;
                }
            }
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        cities = (int **) malloc(n * sizeof(int *));
        for (i = 0; i < n; i++) {
            cities[i] = (int *) malloc(n * sizeof(int));
        }
    }

    for (i = 0; i < n; i++) {
        MPI_Bcast(cities[i], n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    int min_cost = 99999999;
    for (i = rank; i < n; i += size) {
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

    MPI_Reduce(&min_cost, &total_cost, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end_time = MPI_Wtime();

        printf("The minimum cost is %d\n", total_cost);
        printf("Time taken: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
