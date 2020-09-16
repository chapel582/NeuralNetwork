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
		uint32_t NumRows = 32;
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
		printf("MatrixMult m1m2 transpose seconds: %f\n", Seconds);
		
		FreeMatrix(M1);
		FreeMatrix(M2);
		FreeMatrix(AddResult);
	}

	// // SECTION START: MatrixMult: M1 high number of rows test
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 3;
	// 	AllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 3;
	// 	NumColumns = 3;
	// 	AllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	AllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	MatrixMult(MatrixOpJobs, M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixMult: M1 high number of rows test seconds: %f\n", Seconds
	// 	);
		
	// 	FreeMatrix(M1);
	// 	FreeMatrix(M2);
	// 	FreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1 high number of rows test

	// // SECTION START: MatrixMult: M1 high number of columns test
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 3;
	// 	uint32_t NumColumns = 2 << 10;
	// 	AllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 3;
	// 	AllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	AllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	MatrixMult(MatrixOpJobs, M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixMult: M1 high number of columns test seconds: %f\n", Seconds
	// 	);
		
	// 	FreeMatrix(M1);
	// 	FreeMatrix(M2);
	// 	FreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1 high number of columns test

	// // SECTION START: MatrixMult: M1 high rows, columns test
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	AllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 3;
	// 	AllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	AllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	MatrixMult(MatrixOpJobs, M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixMult: M1 high rows, columns test seconds: %f\n", Seconds
	// 	);
		
	// 	FreeMatrix(M1);
	// 	FreeMatrix(M2);
	// 	FreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1 high rows, columns test

	// // SECTION START: MatrixMult: M1,M2 high rows, columns test
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	AllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 2 << 10;
	// 	AllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* MultResult;
	// 	AllocMultResultMatrix(&MultResult, M1, M2);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	MatrixMult(MatrixOpJobs, M1, M2, MultResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixMult: M1 high rows, columns test seconds: %f\n", Seconds
	// 	);
		
	// 	FreeMatrix(M1);
	// 	FreeMatrix(M2);
	// 	FreeMatrix(MultResult);
	// }
	// // SECTION STOP: MatrixMult: M1,M2 high rows, columns test

	// // SECTION START: MatrixAdd: M1 high rows, columns test
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	AllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 2 << 10;
	// 	AllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* AddResult;
	// 	AllocMatrix(&AddResult, M1->NumRows, M1->NumColumns);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	MatrixAdd(MatrixOpJobs, M1, M2, AddResult);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixAdd: M1 high rows, columns test seconds: %f\n", Seconds
	// 	);
		
	// 	FreeMatrix(M1);
	// 	FreeMatrix(M2);
	// 	FreeMatrix(AddResult);
	// }
	// // SECTION STOP: MatrixAdd: M1 high rows, columns test

	// // SECTION START: MatrixAdd: Consecutive, high rows
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 32;
	// 	AllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 32;
	// 	AllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* AddResult;
	// 	AllocMatrix(&AddResult, M1->NumRows, M1->NumColumns);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	for(int Index = 0; Index < 5; Index++)
	// 	{
	// 		MatrixAdd(MatrixOpJobs, M1, M2, AddResult);
	// 		MatrixAdd(MatrixOpJobs, AddResult, M2, M1);
	// 		MatrixAdd(MatrixOpJobs, M1, M2, AddResult);	
	// 	}
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixAdd: Consecutive, high rows seconds: %f\n", Seconds
	// 	);
		
	// 	FreeMatrix(M1);
	// 	FreeMatrix(M2);
	// 	FreeMatrix(AddResult);
	// }
	// // SECTION STOP: MatrixAdd: Consecutive, high rows

	// // SECTION START: MatrixAdd: Consecutive, large data
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 2 << 10;
	// 	uint32_t NumColumns = 2 << 10;
	// 	AllocMatrix(&M1, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M1);		

	// 	matrix* M2;
	// 	NumRows = 2 << 10;
	// 	NumColumns = 2 << 10;
	// 	AllocMatrix(&M2, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M2);

	// 	matrix* AddResult;
	// 	AllocMatrix(&AddResult, M1->NumRows, M1->NumColumns);

	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	for(int Index = 0; Index < 5; Index++)
	// 	{
	// 		MatrixAdd(MatrixOpJobs, M1, M2, AddResult);
	// 		MatrixAdd(MatrixOpJobs, AddResult, M2, M1);
	// 		MatrixAdd(MatrixOpJobs, M1, M2, AddResult);	
	// 	}
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"MatrixAdd: Consecutive, large data seconds: %f\n", Seconds
	// 	);
		
	// 	FreeMatrix(M1);
	// 	FreeMatrix(M2);
	// 	FreeMatrix(AddResult);
	// }
	// // SECTION STOP: MatrixAdd: Consecutive, large data

	// // SECTION START: Dense layer large batch size
	// {
	// 	uint32_t BatchSize = 2 << 10;
	// 	uint32_t InputDim = 4;
	// 	uint32_t OutputDim = 3;
	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	AllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	AllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	DenseForward(MatrixOpJobs, Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large batch: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	AllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	AllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	DenseBack(
	// 		MatrixOpJobs, Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large batch: %f\n", Seconds);
	// 	FreeDenseLayer(DenseLayer);
	// 	FreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large batch size

	// // SECTION START: Dense layer large input dim
	// {
	// 	uint32_t BatchSize = 8;
	// 	uint32_t InputDim = 2 << 10;
	// 	uint32_t OutputDim = 3;
	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	AllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	AllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	DenseForward(MatrixOpJobs, Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large input dim: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	AllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	AllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	DenseBack(
	// 		MatrixOpJobs, Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large input dim: %f\n", Seconds);
	// 	FreeDenseLayer(DenseLayer);
	// 	FreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large input dim

	// // SECTION START: Dense layer large batch and large input dim
	// {
	// 	uint32_t BatchSize = 2 << 10;
	// 	uint32_t InputDim = 2 << 10;
	// 	uint32_t OutputDim = 3;
	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	AllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	AllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	DenseForward(MatrixOpJobs, Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large batch and large input dim: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	AllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	AllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	DenseBack(
	// 		MatrixOpJobs, Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large batch and large input dim: %f\n", Seconds);
	// 	FreeDenseLayer(DenseLayer);
	// 	FreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large batch and large input dim

	// // SECTION START: Dense layer large batch and large input dim and large output dim
	// {
	// 	uint32_t BatchSize = 2 << 10;
	// 	uint32_t InputDim = 2 << 10;
	// 	uint32_t OutputDim = 2 << 10;
	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	AllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	AllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	DenseForward(MatrixOpJobs, Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"DenseForward: large batch, input dim and output dim: %f\n", 
	// 		Seconds
	// 	);

	// 	matrix* NextLayerGradient;
	// 	AllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	AllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	DenseBack(
	// 		MatrixOpJobs, Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf(
	// 		"DenseBack: large batch, input dim, and output dim: %f\n",
	// 		Seconds
	// 	);
	// 	FreeDenseLayer(DenseLayer);
	// 	FreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large batch and large input dim and large output dim

	// // SECTION START: Dense layer large input dim and large output dim
	// {
	// 	uint32_t BatchSize = 32;
	// 	uint32_t InputDim = 2 << 10;
	// 	uint32_t OutputDim = 2 << 10;
	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	AllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	AllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	DenseForward(MatrixOpJobs, Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large input and output dim: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	AllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	AllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	DenseBack(
	// 		MatrixOpJobs, Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large input and output dim: %f\n", Seconds);
	// 	FreeDenseLayer(DenseLayer);
	// 	FreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large input dim and large output dim

	// // SECTION START: Dense layer large output dim
	// {
	// 	uint32_t BatchSize = 32;
	// 	uint32_t InputDim = 4;
	// 	uint32_t OutputDim = 2 << 10;
	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	AllocMatrix(&Outputs, BatchSize, OutputDim);
	// 	MatrixClear(Outputs);

	// 	dense_layer* DenseLayer;
	// 	AllocDenseLayer(&DenseLayer, InputDim, OutputDim);
	// 	FillMatrixConsecutive(&DenseLayer->Weights);
	// 	FillMatrixConsecutive(&DenseLayer->Bias);
	// 	int64_t StartClock = Win32GetWallClock(); 
	// 	DenseForward(MatrixOpJobs, Inputs, DenseLayer, Outputs);
	// 	int64_t EndClock = Win32GetWallClock(); 
	// 	float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseForward: Large output dim: %f\n", Seconds);

	// 	matrix* NextLayerGradient;
	// 	AllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	AllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	StartClock = Win32GetWallClock();
	// 	DenseBack(
	// 		MatrixOpJobs, Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	EndClock = Win32GetWallClock();
	// 	Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 	printf("DenseBack: Large output dim: %f\n", Seconds);
	// 	FreeDenseLayer(DenseLayer);
	// 	FreeDenseLayerTrain(TrainData);
	// }
	// // SECTION STOP: Dense layer large output dim
}