/*
NOTE: build with nvcc. run setup.bat first
can rename file with -o option

Can profile with 
	nvprof .\simple.exe
I had an issue with the profiler when I ran command line not as an admin.
Had an error that mentioned something about permission and users
https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters	
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__
void Add(int N, float* X, float* Y)
{
	int Start = blockIdx.x * blockDim.x + threadIdx.x;
	int Stride = blockDim.x * gridDim.x;
	for(int Index = Start; Index < N; Index += Stride)
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

	int BlockSize = 256;
	int NumBlocks = (N + BlockSize - 1) / BlockSize;
	Add<<<NumBlocks, BlockSize>>>(N, X, Y);
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