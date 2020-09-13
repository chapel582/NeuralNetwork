#include "arg_max.h"

#include "performance.cpp"
#include "matrix.cpp"
#include "neural_net_cpu.cpp"
#include "matrix_test.cpp"
#include "neural_net.cpp"
#include "mnist_test.cpp"

#include <stdio.h>

int main(void)
{
	LARGE_INTEGER PerformanceFrequency;
	QueryPerformanceFrequency(&PerformanceFrequency);
	GlobalPerformanceFrequency = PerformanceFrequency.QuadPart;

	matrix_op_jobs* MatrixOpJobs;
	AllocMatrixOpJobs(&MatrixOpJobs, 4);

	{
		matrix* M1;
		uint32_t NumRows = 32;
		uint32_t NumColumns = 2 << 10;
		AllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 2 << 10;
		NumColumns = 64;
		AllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		AllocMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		MatrixMult(MatrixOpJobs, M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult plain seconds: %f\n", Seconds);

		FreeMatrix(M1);
		FreeMatrix(M2);
		FreeMatrix(MultResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 2 << 10;
		uint32_t NumColumns = 32;
		AllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 2 << 10;
		NumColumns = 64;
		AllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		AllocM1TransposeMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		MatrixMultM1Transpose(MatrixOpJobs, M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult m1 transponse seconds: %f\n", Seconds);

		FreeMatrix(M1);
		FreeMatrix(M2);
		FreeMatrix(MultResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 32;
		uint32_t NumColumns = 2 << 10;
		AllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 64;
		NumColumns = 2 << 10;
		AllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		AllocM2TransposeMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		MatrixMultM2Transpose(MatrixOpJobs, M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult m2 transpose seconds: %f\n", Seconds);
		
		FreeMatrix(M1);
		FreeMatrix(M2);
		FreeMatrix(MultResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 2 << 10;
		uint32_t NumColumns = 32;
		AllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		NumRows = 64;
		NumColumns = 2 << 10;
		AllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* MultResult;
		AllocM1M2TransposeMultResultMatrix(&MultResult, M1, M2);

		int64_t StartClock = Win32GetWallClock(); 
		MatrixMultM1M2Transpose(MatrixOpJobs, M1, M2, MultResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixMult m1m2 transpose seconds: %f\n", Seconds);
		
		FreeMatrix(M1);
		FreeMatrix(M2);
		FreeMatrix(MultResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 64;
		uint32_t NumColumns = 2 << 10;
		AllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		AllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* AddResult;
		AllocMatrix(&AddResult, NumRows, NumColumns);

		int64_t StartClock = Win32GetWallClock(); 
		MatrixAdd(MatrixOpJobs, M1, M2, AddResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixAdd seconds: %f\n", Seconds);
		
		FreeMatrix(M1);
		FreeMatrix(M2);
		FreeMatrix(AddResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 64;
		uint32_t NumColumns = 2 << 10;
		AllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* M2;
		AllocMatrix(&M2, NumRows, NumColumns);
		FillMatrixConsecutive(M2);

		matrix* SubResult;
		AllocMatrix(&SubResult, NumRows, NumColumns);

		int64_t StartClock = Win32GetWallClock(); 
		MatrixSubtract(MatrixOpJobs, M1, M2, SubResult);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixSubtract seconds: %f\n", Seconds);
		
		FreeMatrix(M1);
		FreeMatrix(M2);
		FreeMatrix(SubResult);
	}

	{
		matrix* M1;
		uint32_t NumRows = 64;
		uint32_t NumColumns = 2 << 10;
		AllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);
		
		int64_t StartClock = Win32GetWallClock(); 
		MatrixScalarMult(MatrixOpJobs, 0.5f, M1, M1);
		int64_t EndClock = Win32GetWallClock(); 
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("MatrixScalarMult seconds: %f\n", Seconds);

		FreeMatrix(M1);
	}

	{
		matrix* M1;
		uint32_t NumRows = 64;
		uint32_t NumColumns = 2 << 10;
		AllocMatrix(&M1, NumRows, NumColumns);
		FillMatrixConsecutive(M1);		

		matrix* AddVectorResult;
		AllocMatrix(&AddVectorResult, M1->NumRows, M1->NumColumns);
		matrix* Vector;
		AllocMatrix(&Vector, 1, M1->NumColumns);
		FillMatrixConsecutive(Vector);

		int64_t StartClock = Win32GetWallClock(); 
		AddVectorToRows(MatrixOpJobs, M1, Vector, AddVectorResult);
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
		AllocMatrix(&Inputs, BatchSize, InputDim);
		FillMatrixConsecutive(Inputs);

		matrix* Outputs;
		AllocMatrix(&Outputs, BatchSize, OutputDim);
		MatrixClear(Outputs);

		dense_layer* DenseLayer;
		AllocDenseLayer(&DenseLayer, InputDim, OutputDim);
		FillMatrixConsecutive(&DenseLayer->Weights);
		FillMatrixConsecutive(&DenseLayer->Bias);
		
		int64_t StartClock = Win32GetWallClock();
		DenseForward(MatrixOpJobs, Inputs, DenseLayer, Outputs);
		int64_t EndClock = Win32GetWallClock();
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("Dense forward seconds: %f\n", Seconds);

		matrix* NextLayerGradient;
		AllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
		FillMatrixConsecutive(NextLayerGradient);

		dense_layer_train_data* TrainData;
		AllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
		StartClock = Win32GetWallClock();
		DenseBack(
			MatrixOpJobs, Inputs, NextLayerGradient, DenseLayer, TrainData
		);
		EndClock = Win32GetWallClock();
		Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("Dense back seconds: %f\n", Seconds);
	}
	// SECTION STOP: Dense forward and back performance

// SECTION START: RELU tests
	{
		uint32_t BatchSize = 32;
		uint32_t InputDim = 64;

		matrix* Inputs;
		AllocMatrix(&Inputs, BatchSize, InputDim);
		FillMatrixConsecutive(Inputs);

		matrix* Outputs;
		AllocMatrix(&Outputs, BatchSize, InputDim);
		int64_t StartClock = Win32GetWallClock();
		ReluForward(MatrixOpJobs, Inputs, Outputs);
		int64_t EndClock = Win32GetWallClock();
		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("relu forward seconds: %f\n", Seconds);

		matrix* NextLayerGradient;
		AllocMatrix(&NextLayerGradient, BatchSize, InputDim);
		FillMatrixConsecutive(NextLayerGradient);

		relu_train_data* TrainData;
		AllocReluTrain(&TrainData, BatchSize, InputDim);
		StartClock = Win32GetWallClock();
		ReluBack(MatrixOpJobs, Inputs, NextLayerGradient, TrainData);
		EndClock = Win32GetWallClock();
		Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
		printf("relu back seconds: %f\n", Seconds);
	}
	// SECTION STOP: RELU Tests
}