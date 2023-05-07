#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_CITIES 100000
#define MAX_DISTANCE 100

int main()
{
    int i, j, n, total_cost = 0, cost;
    int **cities; // Declare cities as a pointer to pointer to int
    int visited[MAX_CITIES] = {0};

    printf("Enter the number of cities (max %d): ", MAX_CITIES);
    scanf("%d", &n);

    if (n > MAX_CITIES)
    {
        printf("Error: number of cities exceeds maximum allowed.\n");
        exit(1);
    }

    // Allocate memory for the cities array
    cities = (int **)malloc(n * sizeof(int *));
    for (i = 0; i < n; i++)
    {
        cities[i] = (int *)malloc(n * sizeof(int));
    }

    for (i = 0; i < n; i++)
    {
        for (j = i; j < n; j++)
        {
            if (i == j)
            {
                cities[i][j] = 0;
            }
            else
            {
                cities[i][j] = cities[j][i] = MAX_DISTANCE;
            }
        }
    }

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    int min_cost = 99999999;

    for (i = 0; i < n; i++)
    {
        for (j = i; j < n; j++)
        {
            if (i != j)
            {
                visited[i] = 1;
                visited[j] = 1;

                cost = 0;
                for (int k = 0; k < n; k++)
                {
                    if (!visited[k])
                    {
                        cost += cities[i][k];
                    }
                }

                if (cost < min_cost)
                {
                    min_cost = cost;
                }

                visited[i] = 0;
                visited[j] = 0;
            }
        }
    }

    total_cost = min_cost;

    gettimeofday(&end_time, NULL);

    // Free the memory allocated for the cities array
    for (i = 0; i < n; i++)
    {
        free(cities[i]);
    }
    free(cities);

    printf("The minimum cost is %d\n", total_cost);
    printf("Time taken: %f seconds\n", (end_time.tv_sec - start_time.tv_sec) +
                                           (end_time.tv_usec - start_time.tv_usec) / 1000000.0);

    return 0;
}

