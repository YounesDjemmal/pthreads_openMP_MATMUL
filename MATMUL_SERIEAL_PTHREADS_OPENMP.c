#include <pthread.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#define THREAD_RANGE 8
#define NUM_THREADS  8
#define NUM_AVERAGES 5
#define N 1000
#define M 1000

// COMPILE WITH: gcc -fopenmp matMult-comparison.c -g -o main -lpthread -lm
//     RUN WITH: ./main

int (*A)[N];
int (*B)[N];
int (*C)[N]; // store the serial implementation in this matrix
int (*P)[N]; // store the pthread implementation in this matrix
int (*O)[N]; // store the openmp implementation in this matrix

//Thread info structure will handle the static load balancing through allocation of sets of rows for each thread [startRow .. endRow] to the threads
typedef struct _thread_info
{
    int startRow;
    int endRow;
    int threadNum;
    struct timespec startTime;
    struct timespec endTime;
} thread_info;

//Thread handler function =============================================
void *THandlder(void *param) {
    thread_info *tInfo = (thread_info*)param;
    int start = tInfo->startRow;
    int end = tInfo->endRow;
    for (int i = start; i < end; i++){
        for (int j = 0; j < N; j++){
            P[i][j] = 0;
            for (int k = 0; k < M; k++) {
                P[i][j] +=  A[i][k] * B[k][j];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tInfo->endTime);
}
//=====================================================================

int main(int argc, char *argv[]){

        double pthreadTimings[NUM_THREADS];
        double openMPTimings[NUM_THREADS];


        struct timespec start, finish;
        double elapsed;
        A = malloc(sizeof(int[N][M]));
        B = malloc(sizeof(int[N][M]));
        C = malloc(sizeof(int[N][N])); // serial results
        P = malloc(sizeof(int[N][N])); // parallel results
        O = malloc(sizeof(int[N][N])); // openmp results

        // Initialise our matrix with random data
        for (int i = 0; i < N; i++){
            for (int j = 0; j < M; j++){
                A[i][j] = rand();
                B[i][j] = rand();
            }
        }

        //SERIAL IMPLEMENTATION
        for (int r = 0; r < NUM_AVERAGES; r++) {
            clock_gettime(CLOCK_MONOTONIC, &start);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] = 0;
                    for (int k = 0; k < M; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            clock_gettime(CLOCK_MONOTONIC, &finish);
            elapsed += (finish.tv_sec - start.tv_sec);
            elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        }
        elapsed /= NUM_AVERAGES;
        printf("SERIAL: Took %.9f seconds for %d elements \n", elapsed, N*N);
/*====================================================================================================================================*/
        pthread_t threads[THREAD_RANGE];
        thread_info threadInfo[THREAD_RANGE];

        for (int T = 1; T <= THREAD_RANGE; T++) {
            threadInfo[T].threadNum = T;
            if (N % T == 0 || N < T) { // if the number of rows is evenly divisible by the number of threads, if they're equal or if the number of rows is less than the number of threads
                if (N<T){
                    int numRows = 1;
                }
                else{
                    int numRows = N / T;
                    for (int i = 0; i < T; i++) {
                        threadInfo[i].startRow = i * numRows;
                        threadInfo[i].endRow   = i * numRows + numRows;
                    }
                }
            }
            else {  // if the number of rows in the matrix is NOT evenly divisible by the number of threads
                int rowsRemaining = N;
                int threadsRemaining = T;
                int numRows = 0;
                int thread = 0;
                while (rowsRemaining > 0) {
                    numRows = ceil((float)rowsRemaining/threadsRemaining);
                    threadInfo[thread].startRow = N - rowsRemaining;
                    threadInfo[thread].endRow   = threadInfo[thread].startRow + numRows;
                    rowsRemaining -= numRows;
                    threadsRemaining--;
                    thread++;
                }
            }

            int numIter = N <= T ? N : T;

            for (int r = 0; r < NUM_AVERAGES; r++){
                clock_gettime(CLOCK_MONOTONIC, &start);

                for (long t = 0; t < numIter; t++) {
                    int err = pthread_create(&threads[t], NULL, THandlder, &threadInfo[t]);
                    if (err)
                        exit(-1);
                }
                for (long i = 0; i < numIter; ++i) {
                    pthread_join(threads[i], NULL);
                }
                clock_gettime(CLOCK_MONOTONIC, &finish);
                elapsed = (finish.tv_sec - start.tv_sec);
                elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
                pthreadTimings[T - 1] += elapsed;
            }
        }
        printf("\nPTHREADS: ");
        for (int i = 0; i < THREAD_RANGE; i++) {
            pthreadTimings[i] /= NUM_AVERAGES;
            printf("%.9f ", pthreadTimings[i]);
        }
        printf("\n");

        //Check results
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++)
            {
                assert(C[i][j] == P[i][j]);
            }
        }
/*====================================================================================================================================*/

        //openMP
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int r = 0; r < NUM_AVERAGES; r++) {
            for (int T = 1; T <= THREAD_RANGE; T++) {
                clock_gettime(CLOCK_MONOTONIC, &start);
                omp_set_num_threads(T);
                int i, j, k;
                #pragma omp parallel for private(i, j, k)
                for (i = 0; i < N; i++) {
                    for (j = 0; j < N; j++) {
                        O[i][j] = 0;
                        for (k = 0; k < M; k++) {
                            O[i][j] = O[i][j] + A[i][k] * B[k][j];
                        }
                    }
                }
                clock_gettime(CLOCK_MONOTONIC, &finish);
                elapsed = (finish.tv_sec - start.tv_sec);
                elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
                openMPTimings[T] += elapsed;
            }
        }
        printf("\nOPENMP: ");
        for (int T = 1; T <= THREAD_RANGE; T++) {
            openMPTimings[T]/=NUM_AVERAGES;
            printf("%.9f ", openMPTimings[T]);
        }
        //Make sure computations are correct
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                assert(O[i][j] == P[i][j]);
                assert(O[i][j] == C[i][j]);
            }
        }
        pthread_exit(NULL);
        return 0;
}
