/*
NOTE: build with nvcc. run setup.bat first
can rename file with -o option
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__
void Add(int N, float* X, float* Y)
{
	for(int Index = 0; Index < N; Index++)
	{
		Y[Index] = X[Index] + Y[Index];
	}
}

int main(void)
{
	int N = 1 << 20; // 1M elements
	float* X = NULL; 
	float* Y = NULL;
	cudaMallocManaged(&X, N * sizeof(float));
	cudaMallocManaged(&Y, N * sizeof(float));

	for(int Index = 0; Index < N; Index++)
	{
		X[Index] = 1.0f;
		Y[Index] = 2.0f;
	}

	Add<<<1, 1>>>(N, X, Y);
	cudaDeviceSynchronize();

	float ExpectedValue = 3.0f;
	for(int Index = 0; Index < N; Index++)
	{
		if(Y[Index] != ExpectedValue)
		{
			printf("Y has value %f at %d\n", Y[Index], Index);
		}
	}

	printf("Complete\n");

	return 0;
}