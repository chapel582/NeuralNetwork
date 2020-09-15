// TODO: handle cudaMallocManaged failures
// TODO: query max block size
#include "arg_max.h"
#include "int_shuffler.h"
#include "neural_net.h"
#include "matrix.h"

#include "neural_net_cuda.cu"
#include "performance.cpp"
#include "neural_net.cpp"
#include "matrix.cpp"
#include "mnist_test.cpp"
#include "matrix_test.cpp"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// TODO: combine this with CPU test

int main(void)
{
	CudaInitDeviceProperties(0);

	LARGE_INTEGER PerformanceFrequency;
	QueryPerformanceFrequency(&PerformanceFrequency);
	GlobalPerformanceFrequency = PerformanceFrequency.QuadPart;

	// SECTION START: MatrixMult: M1 low number of rows test	
	{
		matrix* M1;
		uint32_t NumRows = 2 << 4;
		uint32_t NumColumns = 3;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 3;
		NumColumns = 3;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		CudaAllocMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixMult(M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult: M1 low number of rows test seconds: %f\n", Seconds);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(MultResult);
	}
	// SECTION STOP: MatrixMult: M1 low number of rows test

	// SECTION START: MatrixMult: M1 high number of rows test	
	{
		matrix* M1;
		uint32_t NumRows = 2 << 15;
		uint32_t NumColumns = 3;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 3;
		NumColumns = 3;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		CudaAllocMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixMult(M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf(
			"MatrixMult: M1 high number of rows test seconds: %f\n", Seconds
		);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(MultResult);
	}
	// SECTION STOP: MatrixMult: M1 high number of rows test

	// SECTION START: MatrixMult: M1 high number of columns test	
	{
		matrix* M1;
		uint32_t NumRows = 3;
		uint32_t NumColumns = 2 >> 15;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 2 >> 15;
		NumColumns = 3;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		CudaAllocMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixMult(M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf(
			"MatrixMult: M1 high number of columns test seconds: %f\n", Seconds
		);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(MultResult);
	}
	// SECTION STOP: MatrixMult: M1 high number of columns test

}