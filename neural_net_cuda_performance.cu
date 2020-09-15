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
void RoundTripTest(matrix* M1, matrix* M2, matrix* Result)
{	
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;
	if(Start == 0)
	{
		Stride += 1;
	}

	return;
}

__global__
void ConsecutiveAdds(matrix* M1, matrix* M2, matrix* Result, int Iterations)
{	
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;
	for(int Index = 0; Index < Iterations; Index++)
	{
		CudaMatrixAddCore(
			M1, M2, Result, Start, Stride
		);
		__syncthreads();
		CudaMatrixAddCore(Result, M2, M1, Start, Stride);
		__syncthreads();
		CudaMatrixAddCore(M1, M2, Result, Start, Stride);
	}
	return;
}

int main(void)
{
	CudaInitDeviceProperties(0);

	LARGE_INTEGER PerformanceFrequency;
	QueryPerformanceFrequency(&PerformanceFrequency);
	GlobalPerformanceFrequency = PerformanceFrequency.QuadPart;

	// SECTION START: RoundTrip: M1 low number of rows test	
	{
		matrix* M1;
		uint32_t NumRows = 2048;
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

		int BlockSize = GetBlockSize(0);
		int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize, 0);
		int64_t StartClock = Win32GetWallClock(); 
		CudaMatrixMult(M1, M2, MultResult);
		cudaDeviceSynchronize();
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult seconds: %f\n", Seconds);
		
		CudaFreeMatrix(M1);
		CudaFreeMatrix(M2);
		CudaFreeMatrix(MultResult);
	}
	// SECTION STOP: RoundTrip: M1 low number of rows test

	// // SECTION START: RoundTrip: M1, M2 high rows,columns test	
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	CudaAllocMultResultMatrix(&MultResult, M1, M2);

	// 	int BlockSize = GetBlockSize(0);
	// 	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize, 0);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	RoundTripTest<<<NumBlocks, BlockSize>>>(M1, M2, MultResult);
	// 	cudaDeviceSynchronize();
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"RoundTrip: M1,M2 high rows,columns test seconds: %f\n", Seconds
	// 	);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(MultResult);
	// }
	// // SECTION STOP: RoundTrip: M1,M2 high rows,columns test

	// // SECTION START: MatrixMult: M1 low number of rows test	
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 4;
	// 	uint32_t NumColumns = 3;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 3;
	// 	NumColumns = 3;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	CudaAllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaMatrixMult(M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("MatrixMult: M1 low number of rows test seconds: %f\n", Seconds);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1 low number of rows test

	// // SECTION START: MatrixMult: M1 high number of rows test	
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 3;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 3;
	// 	NumColumns = 3;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	CudaAllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaMatrixMult(M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixMult: M1 high number of rows test seconds: %f\n", Seconds
	// 	);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1 high number of rows test

	// // SECTION START: MatrixMult: M1 high number of columns test	
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 3;
	// 	uint32_t NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 3;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	CudaAllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaMatrixMult(M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixMult: M1 high number of columns test seconds: %f\n", Seconds
	// 	);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1 high number of columns test

	// // SECTION START: MatrixMult: M1 high rows,columns test	
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 3;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	CudaAllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaMatrixMult(M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixMult: M1 high rows, columns test seconds: %f\n", Seconds
	// 	);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1 high rows, columns test

	// // SECTION START: MatrixMult: M1, M2 high rows,columns test	
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	CudaAllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaMatrixMult(M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixMult: M1,M2 high rows,columns test seconds: %f\n", Seconds
	// 	);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1,M2 high rows,columns test

	// // SECTION START: MatrixAdd: M1 high rows,columns test	
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* AddResult;
	// 	CudaAllocMatrix(&AddResult, M1->NumRows, M2->NumColumns);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaMatrixAdd(M1, M2, AddResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixAdd: M1 high rows, columns test seconds: %f\n", Seconds
	// 	);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(AddResult);
	// }
	// // SECTION STOP: MatrixAdd: M1 high rows, columns test

	// // SECTION START: MatrixAdd: Consecutive, high rows
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 32;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 32;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* AddResult;
	// 	CudaAllocMatrix(&AddResult, M1->NumRows, M2->NumColumns);

	// 	int BlockSize = GetBlockSize(0);
	// 	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize, 0);
	// 	int64_t StartClock = Win32GetWallClock(); 		
	// 	ConsecutiveAdds<<<NumBlocks, BlockSize>>>(M1, M2, AddResult, 5);
	// 	cudaDeviceSynchronize();
	// 	int64_t EndClock = Win32GetWallClock(); 

	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixAdd: Consecutive, high rows seconds: %f\n", Seconds
	// 	);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(AddResult);
	// }
	// // SECTION STOP: MatrixAdd: Consecutive, high rows

	// // SECTION START: MatrixAdd: Consecutive, large data
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 2 << 10;
	// 	CudaAllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* AddResult;
	// 	CudaAllocMatrix(&AddResult, M1->NumRows, M2->NumColumns);

	// 	int BlockSize = GetBlockSize(0);
	// 	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize, 0);
	// 	int64_t StartClock = Win32GetWallClock(); 		
	// 	ConsecutiveAdds<<<NumBlocks, BlockSize>>>(M1, M2, AddResult, 5);
	// 	cudaDeviceSynchronize();
	// 	int64_t EndClock = Win32GetWallClock(); 

	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixAdd: Consecutive, large data seconds: %f\n", Seconds
	// 	);
		
	// 	CudaFreeMatrix(M1);
	// 	CudaFreeMatrix(M2);
	// 	CudaFreeMatrix(AddResult);
	// }
	// // SECTION STOP: MatrixAdd: Consecutive, large data

	// // SECTION START: Dense layer large batch size
	// {
	// 	uint32_t BatchSize = 2 << 10;
	// 	uint32_t InputDim = 4;
	// 	uint32_t OutputDim = 3;
	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	CudaAllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	CudaAllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaDenseForward(Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large batch: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	CudaAllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	CudaAllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	CudaDenseBack(
	// 		Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large batch: %f\n", Seconds);
	// 	CudaFreeDenseLayer(DenseLayer);
	// 	CudaFreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large batch size

	// // SECTION START: Dense layer large input dim
	// {
	// 	uint32_t BatchSize = 8;
	// 	uint32_t InputDim = 2 << 10;
	// 	uint32_t OutputDim = 3;
	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	CudaAllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	CudaAllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaDenseForward(Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large input dim: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	CudaAllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	CudaAllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	CudaDenseBack(
	// 		Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large input dim: %f\n", Seconds);
	// 	CudaFreeDenseLayer(DenseLayer);
	// 	CudaFreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large input dim

	// // SECTION START: Dense layer large batch and large input dim
	// {
	// 	uint32_t BatchSize = 2 << 10;
	// 	uint32_t InputDim = 2 << 10;
	// 	uint32_t OutputDim = 3;
	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	CudaAllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	CudaAllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaDenseForward(Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large batch and large input dim: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	CudaAllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	CudaAllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	CudaDenseBack(
	// 		Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large batch and large input dim: %f\n", Seconds);
	// 	CudaFreeDenseLayer(DenseLayer);
	// 	CudaFreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large batch and large input dim

	// // SECTION START: Dense layer large batch and large input dim and large output dim
	// {
	// 	uint32_t BatchSize = 2 << 10;
	// 	uint32_t InputDim = 2 << 10;
	// 	uint32_t OutputDim = 2 << 10;
	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	CudaAllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	CudaAllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaDenseForward(Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"DenseForward: large batch, input dim and output dim: %f\n", 
	// 		Seconds
	// 	);

	// 	matrix* NextLayerGradient;
	// 	CudaAllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	CudaAllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	CudaDenseBack(
	// 		Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"DenseBack: large batch, input dim, and output dim: %f\n",
	// 		Seconds
	// 	);
	// 	CudaFreeDenseLayer(DenseLayer);
	// 	CudaFreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large batch and large input dim and large output dim

	// // SECTION START: Dense layer large input dim and large output dim
	// {
	// 	uint32_t BatchSize = 32;
	// 	uint32_t InputDim = 2 << 10;
	// 	uint32_t OutputDim = 2 << 10;
	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	CudaAllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	CudaAllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaDenseForward(Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large input and output dim: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	CudaAllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	CudaAllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	CudaDenseBack(
	// 		Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large input and output dim: %f\n", Seconds);
	// 	CudaFreeDenseLayer(DenseLayer);
	// 	CudaFreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large input dim and large output dim

	// // SECTION START: Dense layer large output dim
	// {
	// 	uint32_t BatchSize = 32;
	// 	uint32_t InputDim = 4;
	// 	uint32_t OutputDim = 2 << 10;
	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	CudaAllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	CudaAllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	CudaDenseForward(Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large output dim: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	CudaAllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	CudaAllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	CudaDenseBack(
	// 		Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large output dim: %f\n", Seconds);
	// 	CudaFreeDenseLayer(DenseLayer);
	// 	CudaFreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large output dim
}