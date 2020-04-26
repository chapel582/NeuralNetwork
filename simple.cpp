/*
NOTE: build with cl. run setup.bat first
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

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
	float* X = (float*) malloc(N * sizeof(float));
	float* Y = (float*) malloc(N * sizeof(float));

	for(int Index = 0; Index < N; Index++)
	{
		X[Index] = 1.0f;
		Y[Index] = 2.0f;
	}

	clock_t Start;
	Start = clock();
	Add(N, X, Y);
	clock_t Clicks = clock() - Start;
	printf("Time to run: %f seconds\n", ((float) Clicks) / CLOCKS_PER_SEC);

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