/*
NOTE: build with nvcc. run setup.bat first
can rename file with -o option

Can profile with 
	nvprof .\simple.exe
I had an issue with the profiler when I ran command line not as an admin.
Had an error that mentioned something about permission and users
	
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

// #include <iostream>
// #include <math.h>
// #include <cuda_profiler_api.h>

// // Kernel function to add the elements of two arrays
// __global__
// void add(int n, float *x, float *y)
// {
//   for (int i = 0; i < n; i++)
//     y[i] = x[i] + y[i];
// }

// int main(void)
// {
//   int N = 1<<20;
//   float *x, *y;

//   // Allocate Unified Memory – accessible from CPU or GPU
//   cudaMallocManaged(&x, N*sizeof(float));
//   cudaMallocManaged(&y, N*sizeof(float));

//   // initialize x and y arrays on the host
//   for (int i = 0; i < N; i++) {
//     x[i] = 1.0f;
//     y[i] = 2.0f;
//   }

//   // Run kernel on 1M elements on the GPU
//   add<<<1, 1>>>(N, x, y);

//   // Wait for GPU to finish before accessing on host
//   cudaDeviceSynchronize();

//   // Check for errors (all values should be 3.0f)
//   float maxError = 0.0f;
//   for (int i = 0; i < N; i++)
//     maxError = fmax(maxError, fabs(y[i]-3.0f));
//   std::cout << "Max error: " << maxError << std::endl;

//   // Free memory
//   cudaFree(x);
//   cudaFree(y);
  
//   return 0;
// }