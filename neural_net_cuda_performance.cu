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

// TODO: combine this test program with CPU test

__global__
void RoundTripTestThread(matrix* M1, matrix* M2, matrix* Result)
{	
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;
	if(Start == 0)
	{
		Stride += 1;
	}

	return;
}

float RoundTripTime(matrix* M1, matrix* M2, matrix* Result)
{
	int BlockSize = GetBlockSize(0);
	int NumBlocks = GetNumBlocks(GetMatrixArrayCount(Result), BlockSize, 0);
	int64_t StartClock = Win32GetWallClock();
	RoundTripTestThread<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
	int64_t EndClock = Win32GetWallClock(); 
	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	
	return Seconds;
}

int main(void)
{
	CudaInitDeviceProperties(0);

	LARGE_INTEGER PerformanceFrequency;
	QueryPerformanceFrequency(&PerformanceFrequency);
	GlobalPerformanceFrequency = PerformanceFrequency.QuadPart;

	{
		int Iterations = 10;
		float Seconds = 0.0f;
		for(int Index = 0; Index < Iterations; Index++)
		{
			matrix* M1;
			uint32_t NumRows = 32;
			uint32_t NumColumns = 2 << 10;
			CudaAllocMatrix(&M1, NumRows, NumColumns);
			FillMatrixConsecutive(M1);		
	
			matrix* M2;
			NumRows = 2 << 10;
			NumColumns = 64;
			CudaAllocMatrix(&M2, NumRows, NumColumns);
			FillMatrixConsecutive(M2);
	
			matrix* MultResult;
			CudaAllocMultResultMatrix(&MultResult, M1, M2);
	
			Seconds += RoundTripTime(M1, M2, MultResult);
			
			CudaFreeMatrix(M1);
			CudaFreeMatrix(M2);
			CudaFreeMatrix(MultResult);
		}
		printf("Average round trip time (mult dims): %f\n", Seconds / Iterations);
	}

	{
		int Iterations = 10;
		float Seconds = 0.0f;
		for(int Index = 0; Index < Iterations; Index++)
		{
			matrix* M1;
			uint32_t NumRows = 64;
			uint32_t NumColumns = 2 << 10;
			CudaAllocMatrix(&M1, NumRows, NumColumns);
			FillMatrixConsecutive(M1);		
	
			matrix* M2;
			CudaAllocMatrix(&M2, NumRows, NumColumns);
			FillMatrixConsecutive(M2);
	
			matrix* Result;
			CudaAllocMatrix(&Result, NumRows, NumColumns);
	
			Seconds += RoundTripTime(M1, M2, Result);
			
			CudaFreeMatrix(M1);
			CudaFreeMatrix(M2);
			CudaFreeMatrix(Result);
		}
		printf(
			"Average round trip time (add/sub dims): %f\n",
			Seconds / Iterations
		);
	}

	{
		int Iterations = 10;
		float Seconds = 0.0f;
		for(int Index = 0; Index < Iterations; Index++)
		{
			matrix* M1;
			uint32_t NumRows = 64;
			uint32_t NumColumns = 2 << 10;
			CudaAllocMatrix(&M1, NumRows, NumColumns);
			FillMatrixConsecutive(M1);		
		
			Seconds += RoundTripTime(M1, M1, M1);
			
			CudaFreeMatrix(M1);
		}
		printf(
			"Average round trip time (scalar mult dims): %f\n",
			Seconds / Iterations
		);
	}

	{
		int Iterations = 10;
		float Seconds = 0.0f;
		for(int Index = 0; Index < Iterations; Index++)
		{
			matrix* M1;
			uint32_t NumRows = 64;
			uint32_t NumColumns = 2 << 10;
			CudaAllocMatrix(&M1, NumRows, NumColumns);
			FillMatrixConsecutive(M1);

			matrix* AddVectorResult;
			CudaAllocMatrix(&AddVectorResult, M1->NumRows, M1->NumColumns);
			matrix* Vector;
			CudaAllocMatrix(&Vector, 1, M1->NumColumns);
			FillMatrixConsecutive(Vector);
		
			Seconds += RoundTripTime(M1, Vector, AddVectorResult);
			
			CudaFreeMatrix(M1);
		}
		printf(
			"Average round trip time (add vector to rows dims): %f\n",
			Seconds / Iterations
		);
	}

	{
		matrix* M1;
		uint32_t NumRows = 32;
		uint32_t NumColumns = 2 << 10;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 2 << 10;
		NumColumns = 64;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		CudaAllocMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixMult(M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult plain seconds: %f\n", Seconds);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(MultResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 2 << 10;
		uint32_t NumColumns = 32;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 2 << 10;
		NumColumns = 64;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		CudaAllocM1TransposeMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixMultM1Transpose(M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult m1 transpose seconds: %f\n", Seconds);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(MultResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 32;
		uint32_t NumColumns = 2 << 10;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 64;
		NumColumns = 2 << 10;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		CudaAllocM2TransposeMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixMultM2Transpose(M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult m2 transpose seconds: %f\n", Seconds);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(MultResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 2 << 10;
		uint32_t NumColumns = 32;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 64;
		NumColumns = 2 << 10;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		CudaAllocM1M2TransposeMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock();
		CudaMatrixMultM1M2Transpose(M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult m1m2 transpose seconds: %f\n", Seconds);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(MultResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 64;
		uint32_t NumColumns = 2 << 10;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* AddResult;
		CudaAllocMatrix(&AddResult, NumRows, NumColumns);

		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixAdd(M1, M2, AddResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixAdd seconds: %f\n", Seconds);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(AddResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 64;
		uint32_t NumColumns = 2 << 10;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		CudaAllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* SubtractResult;
		CudaAllocMatrix(&SubtractResult, NumRows, NumColumns);

		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixSubtract(M1, M2, SubtractResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixSubtract seconds: %f\n", Seconds);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(SubtractResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 64;
		uint32_t NumColumns = 2 << 10;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);
		
		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixScalarMult(0.5f, M1, M1);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixScalarMult seconds: %f\n", Seconds);

		CudaFreeMatrix(M1);
	}

	{
		matrix* M1;
		uint32_t NumRows = 64;
		uint32_t NumColumns = 2 << 10;
		CudaAllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);

		matrix* AddVectorResult;
		CudaAllocMatrix(&AddVectorResult, M1->NumRows, M1->NumColumns);
		matrix* Vector;
		CudaAllocMatrix(&Vector, 1, M1->NumColumns);
		FillMatrixConsecutive(Vector);

		int64_t StartClock = Win32GetWallClock(); 
		CudaAddVectorToRows(M1, Vector, AddVectorResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("AddVectorToRows seconds: %f\n", Seconds);
	}

	// SECTION START: Dense forward and back performance
	{
		uint32_t BatchSize = 32;
		uint32_t InputDim = 2 << 10;
		uint32_t OutputDim = 64;
		matrix* Inputs;
		CudaAllocMatrix(&Inputs, BatchSize, InputDim);
		FillMatrixConsecutive(Inputs);

		matrix* Outputs;
		CudaAllocMatrix(&Outputs, BatchSize, OutputDim);
		MatrixClear(Outputs);

		dense_layer* DenseLayer;
		CudaAllocDenseLayer(&DenseLayer, InputDim, OutputDim);
		FillMatrixConsecutive(&DenseLayer->Weights);
		FillMatrixConsecutive(&DenseLayer->Bias);
		
		int64_t StartClock = Win32GetWallClock();
		CudaDenseForward(Inputs, DenseLayer, Outputs);
		int64_t EndClock = Win32GetWallClock();
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("Dense forward seconds: %f\n", Seconds);

		matrix* NextLayerGradient;
		CudaAllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
		FillMatrixConsecutive(NextLayerGradient);

		dense_layer_train_data* TrainData;
		CudaAllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
		StartClock = Win32GetWallClock();
		CudaDenseBack(
			Inputs, NextLayerGradient, DenseLayer, TrainData
		);
		EndClock = Win32GetWallClock();
		Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("Dense back seconds: %f\n", Seconds);

		CudaFreeDenseLayer(DenseLayer);
		CudaFreeDenseLayerTrain(TrainData);
	}
	// SECTION STOP: Dense forward and back performance

	// SECTION START: RELU tests
	{
		uint32_t BatchSize = 32;
		uint32_t InputDim = 64;

		matrix* Inputs;
		CudaAllocMatrix(&Inputs, BatchSize, InputDim);
		FillMatrixConsecutive(Inputs);

		matrix* Outputs;
		CudaAllocMatrix(&Outputs, BatchSize, InputDim);
		int64_t StartClock = Win32GetWallClock();
		CudaReluForward(Inputs, Outputs);
		int64_t EndClock = Win32GetWallClock();
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("relu forward seconds: %f\n", Seconds);

		matrix* NextLayerGradient;
		CudaAllocMatrix(&NextLayerGradient, BatchSize, InputDim);
		FillMatrixConsecutive(NextLayerGradient);

		relu_train_data* TrainData;
		CudaAllocReluTrain(&TrainData, BatchSize, InputDim);
		StartClock = Win32GetWallClock();
		CudaReluBack(Inputs, NextLayerGradient, TrainData);
		EndClock = Win32GetWallClock();
		Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("relu back seconds: %f\n", Seconds);
	}
	// SECTION STOP: RELU Tests
}