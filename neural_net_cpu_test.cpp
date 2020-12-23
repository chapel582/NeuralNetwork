#include "arg_max.h"
#include "neural_net_common.h"

#include "performance.cpp"
#include "matrix.cpp"
#include "neural_net_cpu.cpp"
#include "matrix_test.cpp"
#include "neural_net.cpp"
#include "mnist_test.cpp"

#include <stdio.h>
#include <shlwapi.h>
#include <float.h>
#pragma fenv_access (on)

#define SAVE_RESULTS 0

#include "tensor.h"

void FillConsecutive(float_tensor* Tensor)
{
	// NOTE: assumes memory is contiguous
	uint32_t TotalElements = GetTotalElements(Tensor);
	for(uint32_t ElementsSet = 0; ElementsSet < TotalElements; ElementsSet++)
	{
		Tensor->Data[ElementsSet] = (float) ElementsSet;
	}
}

matrix* TestMatrixResult(
	matrix* M1,
	char* FilePathBuffer,
	size_t FilePathBufferSize,
	char* TestDataDirectory,
	char* TestName,
	char* EndianString
)
{
	snprintf(
		FilePathBuffer,
		FilePathBufferSize,
		"%s/%s_%s.data",
		TestDataDirectory,
		TestName,
		EndianString
	);
#if SAVE_RESULTS
	SaveMatrix(M1, FilePathBuffer);
#endif

	matrix* CompareTo;
	AllocMatrix(&CompareTo, M1->NumRows, M1->NumColumns);
	bool LoadResult = LoadMatrix(CompareTo, FilePathBuffer);
	if(!LoadResult)
	{
		printf("Could not read %s\n", FilePathBuffer);
	}
	else if(!MatricesAreEquivalent(M1, CompareTo))
	{
		printf("%s failed\n", TestName);
		printf("Expected\n");
		PrintMatrix(CompareTo);
		printf("Got\n");
		PrintMatrix(M1);
	}

	return CompareTo;
}

void TestFloatResult(
	float Result,
	char* FilePathBuffer,
	size_t FilePathBufferSize,
	char* TestDataDirectory,
	char* TestName,
	char* EndianString
)
{
	snprintf(
		FilePathBuffer,
		FilePathBufferSize,
		"%s/%s_%s.data",
		TestDataDirectory,
		TestName,
		EndianString
	);
	FILE* File;
#if SAVE_RESULTS
	fopen_s(&File, FilePathBuffer, "w");
	fwrite(&Result, 1, sizeof(float), File);
	fclose(File);
#endif 
	float Expected;
	fopen_s(&File, FilePathBuffer, "r");
	fread(&Expected, 1, sizeof(float), File);
	fclose(File);

	if(Expected != Result)
	{
		printf("Failure in %s\n", TestName);
	}
}

int main(int argc, char* argv[])
{
	uint32_t OldControl;
	int FpError = _controlfp_s(
		&OldControl, _EM_INEXACT, _MCW_EM
	);
	if(FpError != 0)
	{
		printf("Couldn't enable floating point exceptions\n");
	}

	char TestDataDirectory[260];
	if(argc == 1)
	{
		printf("Assuming test data directory path is ../test_data\n");
		strcpy_s(TestDataDirectory, sizeof(TestDataDirectory), "../test_data");
	}
	else if(argc > 1)
	{
		strcpy_s(TestDataDirectory, sizeof(TestDataDirectory), argv[1]);
		printf("TestDataDirectory is %s\n", TestDataDirectory);
	}
	else
	{
		return -1;
	}

	matrix_op_jobs* MatrixOpJobs;
	AllocMatrixOpJobs(&MatrixOpJobs, 4);

	bool BigEndian = IsBigEndian();
	char EndianString[32];
	if(BigEndian)
	{
		strcpy_s(EndianString, sizeof(EndianString), "BigEndian");
	}
	else
	{
		strcpy_s(EndianString, sizeof(EndianString), "LittleEndian");
	}
	char FilePathBuffer[260];

	LARGE_INTEGER PerformanceFrequency;
	QueryPerformanceFrequency(&PerformanceFrequency);
	GlobalPerformanceFrequency = PerformanceFrequency.QuadPart;

	// SECTION START: Tensor tests
	{
		float_tensor* Tensor = NULL;
		uint32_t Shape[2] = {5, 6};
		AllocAndInitTensor(&Tensor, 2, Shape);
		FillConsecutive(Tensor);
		printf("Full consecutive tensor\n");
		PrintTensor(Tensor);

		uint32_t ZeroIndex = 0;
		float_tensor Slice0 = GetTensor(Tensor, &ZeroIndex, 1);
		printf("Zeroth element of last dimension\n");
		PrintTensor(&Slice0);
		
		uint32_t OneIndex = 1;
		float_tensor Slice1 = GetTensor(Tensor, &OneIndex, 1);
		printf("First element of last dimension\n");
		PrintTensor(&Slice1);

		uint32_t GetElementIndices[2] = {1, 2};
		float_tensor Slice2 = GetTensor(Tensor, GetElementIndices, 2);
		printf("Zero-dimensional tensor\n");
		PrintTensor(&Slice2);

		float_tensor Transposed2d = Transpose(Tensor, 0, 1);
		printf("Transposed2d\n");
		PrintTensor(&Transposed2d);

		float_tensor Slice0OfTransposed = GetTensor(
			&Transposed2d, &ZeroIndex, 1
		);
		printf("Zeroth slice of transposed\n");
		PrintTensor(&Slice0OfTransposed);

		float_tensor Slice1OfTransposed = GetTensor(
			&Transposed2d, &OneIndex, 1
		);
		printf("First slice of transposed\n");
		PrintTensor(&Slice1OfTransposed);

		float ScalarFromTranspose = GetElement(
			&Transposed2d, GetElementIndices, 2
		);
		printf("Scalar from transposed 2d tensor\n");
		printf("%f\n", ScalarFromTranspose);

		uint32_t Pairs[4] = {0, 2, 1, 3};
		float_tensor SliceFrom2d = Slice(
			Tensor, Pairs, ARRAY_COUNT(Pairs)
		);
		printf("[0:2][1:3] slice from consecutive\n");
		PrintTensor(&SliceFrom2d);

		uint32_t Pairs2[4] = {1, 4, 2, 5};
		SliceFrom2d = Slice(
			Tensor, Pairs2, ARRAY_COUNT(Pairs2)
		);
		printf("[1:4][2:5] slice from consecutive\n");
		PrintTensor(&SliceFrom2d);

		uint32_t Pairs3[4] = {1, 3, 1, 3};
		float_tensor SliceFromSlice = Slice(
			&SliceFrom2d, Pairs3, ARRAY_COUNT(Pairs3)
		);
		printf("[1:3][1:3] slice from slice\n");
		PrintTensor(&SliceFromSlice);

		SliceFrom2d = Slice(
			&Transposed2d, Pairs2, ARRAY_COUNT(Pairs2)
		);
		printf("[1:4][2:5] slice from transpose\n");
		PrintTensor(&SliceFrom2d);
		// TODO: test free
	}

	{
		float_tensor* ThreeDTensor = NULL;
		uint32_t Shape[3] = {5, 6, 7};
		AllocAndInitTensor(&ThreeDTensor, 3, Shape);
		FillConsecutive(ThreeDTensor);
		printf("Full consecutive 3D tensor\n");
		PrintTensor(ThreeDTensor);
		
		uint32_t ZeroIndex = 0;
		float_tensor TwoDZeroth = GetTensor(ThreeDTensor, &ZeroIndex, 1);
		printf("Zeroth 2D tensor\n");
		PrintTensor(&TwoDZeroth);
		
		uint32_t OneIndex = 1;
		float_tensor TwoDFirst = GetTensor(ThreeDTensor, &OneIndex, 1);
		printf("First 2d tensor\n");
		PrintTensor(&TwoDFirst);

		uint32_t GetOneD[2] = {1, 2};
		float_tensor OneDTensor = GetTensor(ThreeDTensor, GetOneD, 2);
		printf("One-dimensional tensor\n");
		PrintTensor(&OneDTensor);

		uint32_t GetElementIndices[3] = {1, 2, 3};
		float_tensor ZeroD = GetTensor(ThreeDTensor, GetElementIndices, 3);
		printf("Zero-dimensional tensor\n");
		PrintTensor(&ZeroD);

		uint32_t ZeroDFromOneDIndex = 3;
		float_tensor ZeroDFromOneD = GetTensor(
			&OneDTensor, &ZeroDFromOneDIndex, 1
		);
		printf("Zero-dimensional tensor from one-dimensional tensor\n");
		PrintTensor(&ZeroDFromOneD);

		float Scalar = GetElement(ThreeDTensor, GetElementIndices, 3);
		printf("Scalar from 3d tensor\n");
		printf("%f\n", Scalar);

		Scalar = GetElement(ThreeDTensor, 1, 2, 3);
		printf("Scalar from 3d tensor using variable args\n");
		printf("%f\n", Scalar);

		float_tensor Transposed3d = Transpose(ThreeDTensor, 0, 1);
		printf("Transposed 3d tensor\n");
		PrintTensor(&Transposed3d);

		float ScalarFromTranspose = GetElement(
			&Transposed3d, GetElementIndices, 3
		);
		printf("Scalar from transposed 3d tensor\n");
		printf("%f\n", ScalarFromTranspose);

		float_tensor Slice0OfTransposed = GetTensor(
			&Transposed3d, &ZeroIndex, 1
		);
		printf("Zeroth slice of transposed\n");
		PrintTensor(&Slice0OfTransposed);

		float_tensor Slice1OfTransposed = GetTensor(
			&Transposed3d, &OneIndex, 1
		);
		printf("First slice of transposed\n");
		PrintTensor(&Slice1OfTransposed);

		uint32_t Pairs[6] = {0, 1, 0, 2, 0, 2};
		float_tensor SliceFrom3d = Slice(
			ThreeDTensor, Pairs, ARRAY_COUNT(Pairs)
		);
		printf("[0:1][0:2][0:2] slice from consecutive\n");
		PrintTensor(&SliceFrom3d);

		uint32_t Pairs2[6] = {0, 2, 0, 2, 0, 2};
		SliceFrom3d = Slice(
			ThreeDTensor, Pairs2, ARRAY_COUNT(Pairs2)
		);
		printf("[0:2][0:2][0:2] slice from consecutive\n");
		PrintTensor(&SliceFrom3d);

		uint32_t Pairs3[6] = {0, 2, 1, 3, 2, 4};
		SliceFrom3d = Slice(
			ThreeDTensor, Pairs3, ARRAY_COUNT(Pairs3)
		);
		printf("[0:2][1:3][2:4] slice from consecutive\n");
		PrintTensor(&SliceFrom3d);
		// TODO: test free
	}

	// // SECTION START: Matrix tests
	// {
	// 	matrix* M1;
	// 	uint32_t NumRows = 3;
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
	// 	MatrixMultCore(M1, M2, MultResult, 0, 1);
	// 	// NOTE: TestMatrixResult returns a matrix pointer that can be freed
	// 	TestMatrixResult(
	// 		MultResult,
	// 		FilePathBuffer,
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixMultCore",
	// 		EndianString
	// 	);

	// 	MatrixClear(MultResult);
	// 	TestMatrixResult(
	// 		MultResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixClear",
	// 		EndianString
	// 	);

	// 	MatrixMult(MatrixOpJobs, M1, M2, MultResult);
	// 	TestMatrixResult(
	// 		MultResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixMult",
	// 		EndianString
	// 	);

	// 	matrix* M3;
	// 	NumRows = 3;
	// 	NumColumns = 2;
	// 	AllocMatrix(&M3, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M3);

	// 	matrix* M4;
	// 	NumRows = 2;
	// 	NumColumns = 3;
	// 	AllocMatrix(&M4, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M4);

	// 	matrix* MultResult2;
	// 	AllocMultResultMatrix(&MultResult2, M3, M4);
	// 	MatrixMult(MatrixOpJobs, M3, M4, MultResult2);
	// 	TestMatrixResult(
	// 		MultResult2,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"NonSquareMatrixMult",
	// 		EndianString
	// 	);

	// 	matrix* AddResult;
	// 	AllocMatrix(&AddResult, M1->NumRows, M1->NumColumns);
	// 	MatrixAdd(MatrixOpJobs, M1, M2, AddResult);
	// 	TestMatrixResult(
	// 		AddResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixAdd",
	// 		EndianString
	// 	);

	// 	matrix* AddVectorResult;
	// 	AllocMatrix(&AddVectorResult, M1->NumRows, M1->NumColumns);
	// 	matrix* Vector;
	// 	AllocMatrix(&Vector, 1, M1->NumColumns);
	// 	FillMatrixConsecutive(Vector);
	// 	AddVectorToRows(MatrixOpJobs, M1, Vector, AddVectorResult);
	// 	TestMatrixResult(
	// 		AddVectorResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"AddVectorToRows",
	// 		EndianString
	// 	);

	// 	matrix* M5;
	// 	NumRows = 2;
	// 	NumColumns = 3;
	// 	AllocMatrix(&M5, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M5);

	// 	matrix* M6;
	// 	NumRows = 2;
	// 	NumColumns = 3;
	// 	AllocMatrix(&M6, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M6);

	// 	matrix* M5TMultResult;
	// 	AllocM1TransposeMultResultMatrix(&M5TMultResult, M5, M6);

	// 	MatrixMultM1TransposeCore(M5, M6, M5TMultResult, 0, 1);

	// 	MatrixClear(M5TMultResult);
	// 	MatrixMultM1Transpose(MatrixOpJobs, M5, M6, M5TMultResult);
	// 	TestMatrixResult(
	// 		M5TMultResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixMultM1Transpose",
	// 		EndianString
	// 	);

	// 	MatrixClear(M5TMultResult);
	// 	SetMatrixElement(M6, 0, 1, 7);
	// 	SetMatrixElement(M6, 1, 2, 13);
		
	// 	MatrixMultM1Transpose(MatrixOpJobs, M5, M6, M5TMultResult);
	// 	TestMatrixResult(
	// 		M5TMultResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"NonSymmetricMatrixMultM1Transpose",
	// 		EndianString
	// 	);

	// 	matrix* M6TMultResult;
	// 	AllocM2TransposeMultResultMatrix(&M6TMultResult, M5, M6);

	// 	MatrixMultM2TransposeCore(M5, M6, M6TMultResult, 0, 1);
	// 	TestMatrixResult(
	// 		M6TMultResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixMultM2TransposeCore",
	// 		EndianString
	// 	);

	// 	MatrixClear(M6TMultResult);
	// 	MatrixMultM2Transpose(MatrixOpJobs, M5, M6, M6TMultResult);
	// 	TestMatrixResult(
	// 		M6TMultResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixMultM2Transpose",
	// 		EndianString
	// 	);

	// 	matrix* M7;
	// 	NumRows = 2;
	// 	NumColumns = 3;
	// 	AllocMatrix(&M7, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M7);

	// 	matrix* M8;
	// 	NumRows = 3;
	// 	NumColumns = 2;
	// 	AllocMatrix(&M8, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M8);

	// 	matrix* M7TM8TMultResult;
	// 	AllocM1M2TransposeMultResultMatrix(&M7TM8TMultResult, M7, M8);

	// 	MatrixMultM1M2TransposeCore(M7, M8, M7TM8TMultResult, 0, 1);
	// 	TestMatrixResult(
	// 		M7TM8TMultResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixMultM1M2TransposeCore",
	// 		EndianString
	// 	);

	// 	MatrixClear(M7TM8TMultResult);
	// 	MatrixMultM1M2Transpose(MatrixOpJobs, M7, M8, M7TM8TMultResult);
	// 	TestMatrixResult(
	// 		M7TM8TMultResult,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixMultM1M2Transpose",
	// 		EndianString
	// 	);

	// 	matrix* M9;
	// 	NumRows = 3;
	// 	NumColumns = 4;
	// 	AllocMatrix(&M9, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M9);
		
	// 	MatrixScalarMult(MatrixOpJobs, 0.5f, M9, M9);
	// 	TestMatrixResult(
	// 		M9,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixScalarMult",
	// 		EndianString
	// 	);

	// 	matrix* M10;
	// 	NumRows = 4;
	// 	NumColumns = 4;
	// 	AllocMatrix(&M10, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M10);

	// 	matrix* M10Mean;
	// 	AllocMatrixMeanResult(&M10Mean, M10);
	// 	MatrixMeanCore(M10, M10Mean, 0, 1);
	// 	TestMatrixResult(
	// 		M10Mean,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixRowMeanCore",
	// 		EndianString
	// 	);

	// 	MatrixClear(M10Mean);
	// 	MatrixMean(MatrixOpJobs, M10, M10Mean);
	// 	TestMatrixResult(
	// 		M10Mean,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixRowMean",
	// 		EndianString
	// 	);

	// 	matrix* M11;
	// 	NumRows = 3;
	// 	NumColumns = 4;
	// 	AllocMatrix(&M11, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M11);

	// 	matrix* M12;
	// 	AllocMatrix(&M12, NumRows, NumColumns);
	// 	FillMatrixConsecutive(M12);
	// 	SetMatrixElement(M12, 0, 0, -2.0f);
	// 	matrix* SubResult;
	// 	AllocMatrix(&SubResult, NumRows, NumColumns);
	// 	MatrixSubtract(MatrixOpJobs, M11, M12, SubResult);
	// 	TestMatrixResult(
	// 		SubResult,
	// 		FilePathBuffer,
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MatrixSub",
	// 		EndianString
	// 	);
	// 	// NOTE: if memory starts getting hefty, free memory here
	// }
	// // SECTION STOP: Matrix tests

	// // SECTION START: Dense layer tests
	// {
	// 	uint32_t BatchSize = 8;
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
	// 	DenseForward(MatrixOpJobs, Inputs, DenseLayer, Outputs);

	// 	TestMatrixResult(
	// 		Outputs,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"ForwardDense",
	// 		EndianString
	// 	);

	// 	matrix* NextLayerGradient;
	// 	AllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	dense_layer_train_data* TrainData;
	// 	AllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
	// 	DenseBack(
	// 		MatrixOpJobs, Inputs, NextLayerGradient, DenseLayer, TrainData
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Weights,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"DenseWeightsAfterUpdate",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Bias,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"DenseBiasAfterUpdate",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&TrainData->LayerGradient,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"DenseLayerGradient",
	// 		EndianString
	// 	);

	// 	{
	// 		matrix* Destination;
	// 		matrix* Source;
	// 		AllocMatrix(&Source, 4, 4);
	// 		AllocMatrix(&Destination, 4, 4);
	// 		FillMatrixConsecutive(Source);
	// 		CopyMatrix(
	// 			MatrixOpJobs,
	// 			Destination,
	// 			Source
	// 		);
	// 		TestMatrixResult(
	// 			Destination,
	// 			FilePathBuffer, 
	// 			sizeof(FilePathBuffer),
	// 			TestDataDirectory,
	// 			"CopySymmetricMatrix",
	// 			EndianString
	// 		);
	// 	}

	// 	{
	// 		matrix* Destination;
	// 		matrix* Source;
	// 		AllocMatrix(&Source, 4, 3);
	// 		AllocMatrix(&Destination, 4, 3);
	// 		FillMatrixConsecutive(Source);
	// 		CopyMatrix(
	// 			MatrixOpJobs,
	// 			Destination,
	// 			Source
	// 		);
	// 		TestMatrixResult(
	// 			Destination,
	// 			FilePathBuffer, 
	// 			sizeof(FilePathBuffer),
	// 			TestDataDirectory,
	// 			"CopyAsymmetricMatrix",
	// 			EndianString
	// 		);
	// 	}

	// 	{
	// 		matrix* IdentityMatrix;
	// 		AllocMatrix(&IdentityMatrix, 5, 5);
	// 		FillIdentityMatrix(IdentityMatrix);
	// 		TestMatrixResult(
	// 			IdentityMatrix,
	// 			FilePathBuffer, 
	// 			sizeof(FilePathBuffer),
	// 			TestDataDirectory,
	// 			"IdentityMatrix",
	// 			EndianString
	// 		);
	// 	}
	// }
	// // SECTION STOP: Dense layer tests

	// // SECTION START: RELU tests
	// {
	// 	uint32_t BatchSize = 8;
	// 	uint32_t InputDim = 4;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	matrix* Outputs;
	// 	AllocMatrix(&Outputs, BatchSize, InputDim);
	// 	ReluForward(MatrixOpJobs, Inputs, Outputs);
	// 	TestMatrixResult(
	// 		Outputs,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"ReluForwardPositive",
	// 		EndianString
	// 	);

	// 	matrix* NextLayerGradient;
	// 	AllocMatrix(&NextLayerGradient, BatchSize, InputDim);
	// 	FillMatrixConsecutive(NextLayerGradient);

	// 	relu_train_data* TrainData;
	// 	AllocReluTrain(&TrainData, BatchSize, InputDim);
	// 	ReluBack(MatrixOpJobs, Inputs, NextLayerGradient, TrainData);
	// 	TestMatrixResult(
	// 		&TrainData->LayerGradient,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"ReluLayerGradientPositive",
	// 		EndianString
	// 	);

	// 	MatrixScalarMult(MatrixOpJobs, -1.0f, Inputs, Inputs);
	// 	ReluForward(MatrixOpJobs, Inputs, Outputs);
	// 	TestMatrixResult(
	// 		Outputs,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"ReluForwardNegative",
	// 		EndianString
	// 	);

	// 	ReluBack(MatrixOpJobs, Inputs, NextLayerGradient, TrainData);
	// 	TestMatrixResult(
	// 		&TrainData->LayerGradient,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"ReluLayerGradientNegative",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: RELU Tests

	// // SECTION START: Softmax & Cross-Entropy Loss Tests
	// {
	// 	uint32_t BatchSize = 8;
	// 	uint32_t NumClasses = 4;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, NumClasses);
	// 	FillMatrixConsecutive(Inputs);
	// 	SetMatrixElement(Inputs, 0, 0, -1.0f);
	// 	SetMatrixElement(Inputs, BatchSize - 1, NumClasses - 1, 20.0f);

	// 	matrix* Predictions = NULL;
	// 	AllocMatrix(&Predictions, BatchSize, NumClasses);

	// 	softmax_layer* SoftmaxLayer;
	// 	AllocSoftmaxLayer(&SoftmaxLayer, BatchSize, NumClasses);
	// 	SoftmaxForward(
	// 		MatrixOpJobs,
	// 		SoftmaxLayer,
	// 		Inputs,
	// 		Predictions
	// 	);
	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"SoftmaxForward",
	// 		EndianString
	// 	);
		
	// 	matrix* Labels; 
	// 	AllocMatrix(&Labels, BatchSize, NumClasses);
	// 	FillOneHotMatrix(Labels);
		
	// 	float Loss = CrossEntropyForward(MatrixOpJobs, Predictions, Labels);
	// 	TestFloatResult(
	// 		Loss,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"CrossEntropyForwardLoss",
	// 		EndianString
	// 	);
		
	// 	// TODO: test softmax cross-entropy
	// 	// cross_entropy_softmax_train_data* TrainData;
	// 	// AllocCrossEntropySoftmaxTrain(&TrainData, SoftmaxLayer);
	// 	// SoftmaxCrossEntropyBack(MatrixOpJobs, Predictions, Labels, TrainData);
	// 	// TestMatrixResult(
	// 	// 	&TrainData->LayerGradient,
	// 	// 	FilePathBuffer, 
	// 	// 	sizeof(FilePathBuffer),
	// 	// 	TestDataDirectory,
	// 	// 	"CrossEntropySoftmaxBack",
	// 	// 	EndianString
	// 	// );

	// 	// NOTE: high value softmax imput test
	// 	MatrixScalarMult(MatrixOpJobs, 20.0f, Inputs, Inputs);
	// 	SoftmaxForward(MatrixOpJobs, SoftmaxLayer, Inputs, Predictions);
	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"LargeInputSoftmaxForward",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Softmax & Cross-Entropy Loss Tests

	// // SECTION START: MSE Test
	// {
	// 	uint32_t BatchSize = 8;
	// 	uint32_t NumClasses = 4;

	// 	matrix* Predictions = NULL;
	// 	AllocMatrix(&Predictions, BatchSize, NumClasses);
	// 	FillOneHotMatrix(Predictions);
		
	// 	matrix* Labels; 
	// 	AllocMatrix(&Labels, BatchSize, NumClasses);
	// 	FillOneHotMatrix(Labels);

	// 	float Loss = MseForward(MatrixOpJobs, Predictions, Labels);
	// 	TestFloatResult(
	// 		Loss,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MSELoss",
	// 		EndianString
	// 	);

	// 	mse_train_data* TrainData;
	// 	AllocMseTrainData(&TrainData, BatchSize, NumClasses);
	// 	MeanSquaredBack(MatrixOpJobs, Predictions, Labels, TrainData);
	// 	TestMatrixResult(
	// 		&TrainData->LayerGradient,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"MSEBackOK",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: MSE Test

	// // SECTION START: Linear NN test
	// {
	// 	uint32_t BatchSize = 10;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		1
	// 	);
	// 	AddDense(NeuralNet, 1);
	// 	AddDense(NeuralNet, 1);
	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	SetMatrixElement(&DenseLayer->Weights, 0, 0, 2);
	// 	SetMatrixElement(&DenseLayer->Bias, 0, 0, 1);
	// 	DenseLayer = (dense_layer*) NeuralNet->LastLink->Data;
	// 	SetMatrixElement(&DenseLayer->Weights, 0, 0, 3);
	// 	SetMatrixElement(&DenseLayer->Bias, 0, 0, 1);
	// 	// NOTE: should be equivalent to 6x + 4

	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"LinearForwardNN",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Linear NN test

	// // SECTION START: Dim loss NN test
	// {
	// 	uint32_t BatchSize = 4;
	// 	uint32_t InputDim = 4;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		1
	// 	);
	// 	AddDense(NeuralNet, 2);
	// 	AddDense(NeuralNet, 1);
	// 	dense_layer* DenseLayer1 = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	FillMatrixConsecutive(&DenseLayer1->Weights);
	// 	FillMatrixConsecutive(&DenseLayer1->Bias);
	// 	dense_layer* DenseLayer2 = (dense_layer*) NeuralNet->LastLink->Data;
	// 	FillMatrixConsecutive(&DenseLayer2->Weights);
	// 	FillMatrixConsecutive(&DenseLayer2->Bias);

	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"DimReductionLinearForwardNN",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Dim loss NN test

	// // SECTION START: Positive Relu NN test
	// {
	// 	uint32_t BatchSize = 10;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		1
	// 	);
	// 	AddDense(NeuralNet, 1);
	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	SetMatrixElement(&DenseLayer->Weights, 0, 0, 2);
	// 	SetMatrixElement(&DenseLayer->Bias, 0, 0, 1);
	// 	AddRelu(NeuralNet);
	// 	// NOTE: should be equivalent to 2x + 1

	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"PosReluNN",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Positive Relu NN test

	// // SECTION START: Negative Relu NN test
	// {
	// 	uint32_t BatchSize = 10;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixNegativeConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		1
	// 	);
	// 	AddDense(NeuralNet, 1);
	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	SetMatrixElement(&DenseLayer->Weights, 0, 0, 2);
	// 	SetMatrixElement(&DenseLayer->Bias, 0, 0, 1);
	// 	AddRelu(NeuralNet);
	// 	// NOTE: should be equivalent to 2x, but then everything is zeroed

	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"NegReluNN",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Negative Relu NN test

	// // TODO: add this test back in once we get softmax working well
	// // // SECTION START: Softmax Relu NN test
	// // {
	// // 	uint32_t BatchSize = 8;
	// // 	uint32_t InputDim = 2;

	// // 	matrix* Inputs;
	// // 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// // 	FillMatrixConsecutive(Inputs);

	// // 	neural_net* NeuralNet = NULL;
	// // 	AllocNeuralNet(
	// // 		&NeuralNet,
	// // 		BatchSize,
	// // 		InputDim,
	// // 		1
	// // 	);
	// // 	AddDense(NeuralNet, 4);
	// // 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// // 	FillOneHotMatrix(&DenseLayer->Weights);
	// // 	AddSoftmax(NeuralNet);

	// // 	matrix* Predictions = NULL;
	// // 	NeuralNetForward(
	// // 		NeuralNet,
	// // 		Inputs,
	// // 		NULL,
	// // 		&Predictions,
	// // 		NULL
	// // 	);

	// // 	TestMatrixResult(
	// // 		Predictions,
	// // 		FilePathBuffer, 
	// // 		sizeof(FilePathBuffer),
	// // 		TestDataDirectory,
	// // 		"SoftmaxNN",
	// // 		EndianString
	// // 	);
	// // }
	// // // SECTION STOP: Softmax Relu NN test

	// // SECTION START: One neuron training
	// {
	// 	uint32_t BatchSize = 5;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		1
	// 	);
	// 	AddDense(NeuralNet, 1);

	// 	// NOTE: should be equivalent to 2x + 1
	// 	matrix* Labels;
	// 	AllocMatrix(&Labels, BatchSize, 1);
	// 	SetMatrixElement(Labels, 0, 0, 3);
	// 	SetMatrixElement(Labels, 1, 0, 5);
	// 	SetMatrixElement(Labels, 2, 0, 7);
	// 	SetMatrixElement(Labels, 3, 0, 9);
	// 	SetMatrixElement(Labels, 4, 0, 11);

	// 	neural_net_trainer* Trainer;
	// 	AllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	TrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100
	// 	);

	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;

	// 	TestMatrixResult(
	// 		&DenseLayer->Weights,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"OneNeuronNN_Weights",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Bias,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"OneNeuronNN_Bias",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: One neuron training

	// // SECTION START: More one neuron training
	// {
	// 	uint32_t BatchSize = 6;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		1
	// 	);
	// 	AddDense(NeuralNet, 1);

	// 	// NOTE: should be equivalent to 5x - 3
	// 	matrix* Labels;
	// 	AllocMatrix(&Labels, BatchSize, 1);
	// 	SetMatrixElement(Labels, 0, 0, 2);
	// 	SetMatrixElement(Labels, 1, 0, 7);
	// 	SetMatrixElement(Labels, 2, 0, 12);
	// 	SetMatrixElement(Labels, 3, 0, 17);
	// 	SetMatrixElement(Labels, 4, 0, 22);
	// 	SetMatrixElement(Labels, 5, 0, 27);

	// 	neural_net_trainer* Trainer;
	// 	AllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	TrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100
	// 	);

	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;

	// 	TestMatrixResult(
	// 		&DenseLayer->Weights,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"OneNeuronNN_Weights_2",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Bias,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"OneNeuronNN_Bias_2",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: More one neuron training

	// // SECTION START: threaded one neuron training
	// {
	// 	uint32_t BatchSize = 6;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		4
	// 	);
	// 	AddDense(NeuralNet, 1);

	// 	// NOTE: should be equivalent to 5x - 3
	// 	matrix* Labels;
	// 	AllocMatrix(&Labels, BatchSize, 1);
	// 	SetMatrixElement(Labels, 0, 0, 2);
	// 	SetMatrixElement(Labels, 1, 0, 7);
	// 	SetMatrixElement(Labels, 2, 0, 12);
	// 	SetMatrixElement(Labels, 3, 0, 17);
	// 	SetMatrixElement(Labels, 4, 0, 22);
	// 	SetMatrixElement(Labels, 5, 0, 27);

	// 	neural_net_trainer* Trainer;
	// 	AllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	TrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100
	// 	);

	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;

	// 	TestMatrixResult(
	// 		&DenseLayer->Weights,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"OneNeuronNN_Weights_threaded",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Bias,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"OneNeuronNN_Bias_threaded",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: threaded one neuron training

	// // SECTION START: threaded two neuron training
	// {
	// 	uint32_t BatchSize = 2;
	// 	uint32_t InputDim = 2;
	// 	uint32_t OutputDim = 2;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		4
	// 	);
	// 	AddDense(NeuralNet, OutputDim);

	// 	// NOTE: Labels set up to converge weight to 
	// 	/* CONT: 
	// 		W = 
	// 			| 2 3 |
	// 			| 4 5 |
	// 		b = 
	// 			| 1 2 |
	// 	*/
	// 	matrix* Labels;
	// 	AllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	SetMatrixElement(Labels, 0, 0, 11);
	// 	SetMatrixElement(Labels, 0, 1, 15);

	// 	SetMatrixElement(Labels, 1, 0, 23);
	// 	SetMatrixElement(Labels, 1, 1, 31);

	// 	neural_net_trainer* Trainer;
	// 	AllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	TrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100
	// 	);

	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;

	// 	TestMatrixResult(
	// 		&DenseLayer->Weights,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"TwoNeuronNN_Weights_threaded",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Bias,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"TwoNeuronNN_Bias_threaded",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: threaded two neuron training

	// // SECTION START: threaded one layer training and prediction
	// {
	// 	uint32_t BatchSize = 2;
	// 	uint32_t InputDim = 2;
	// 	uint32_t OutputDim = 2;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		4
	// 	);
	// 	AddDense(NeuralNet, OutputDim);

	// 	// NOTE: Labels set up to converge weight to 
	// 	/* CONT: 
	// 		W = 
	// 			| 2 3 |
	// 			| 4 5 |
	// 		b = 
	// 			| 1 2 |
	// 	*/
	// 	matrix* Labels;
	// 	AllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	SetMatrixElement(Labels, 0, 0, 11);
	// 	SetMatrixElement(Labels, 0, 1, 15);

	// 	SetMatrixElement(Labels, 1, 0, 23);
	// 	SetMatrixElement(Labels, 1, 1, 31);

	// 	neural_net_trainer* Trainer;
	// 	AllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	TrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100
	// 	);

	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"LinearOneLayerPrediction",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: threaded one layer training and prediction

	// // SECTION START: threaded one layer training and prediction
	// {
	// 	uint32_t BatchSize = 2;
	// 	uint32_t InputDim = 2;
	// 	uint32_t HiddenDim = 32;
	// 	uint32_t OutputDim = 2;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		4
	// 	);
	// 	AddDense(NeuralNet, HiddenDim);
	// 	AddDense(NeuralNet, OutputDim);

	// 	matrix* Labels;
	// 	AllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	SetMatrixElement(Labels, 0, 0, 11);
	// 	SetMatrixElement(Labels, 0, 1, 15);

	// 	SetMatrixElement(Labels, 1, 0, 23);
	// 	SetMatrixElement(Labels, 1, 1, 31);

	// 	neural_net_trainer* Trainer;
	// 	AllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	TrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100
	// 	);

	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"LinearTwoLayerPrediction",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: threaded one layer training and prediction

	// // SECTION START: XOR Forward
	// {
	// 	uint32_t BatchSize = 4;
	// 	uint32_t InputDim = 2;
	// 	uint32_t HiddenDim = 8;
	// 	uint32_t OutputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	SetMatrixElement(Inputs, 0, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 0, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 1, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 1, 1, 1.0f);

	// 	SetMatrixElement(Inputs, 2, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 2, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 3, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 3, 1, 1.0f);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		4
	// 	);
	// 	AddDense(NeuralNet, HiddenDim);
	// 	AddRelu(NeuralNet);
	// 	AddDense(NeuralNet, OutputDim);

	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	float FirstLayerWeights[] = {
	// 		-0.6014779f,
	// 		0.6651714f,
	// 		-0.33612493f,
	// 		0.7284934f,
	// 		0.49762666f,
	// 		-0.33008203f,
	// 		-0.66281337f,
	// 		-0.7124146f,
	// 		0.6431314f,
	// 		0.6383204f,
	// 		-0.44230828f,
	// 		-0.7284539f,
	// 		0.7563155f,
	// 		0.29757506f,
	// 		-0.2068302f,
	// 		-0.04522699f
	// 	};
	// 	memcpy(
	// 		DenseLayer->Weights.Data,
	// 		FirstLayerWeights,
	// 		sizeof(FirstLayerWeights)
	// 	);
	// 	float FirstLayerBias[] = {
	// 		-3.8563503e-06f,
	// 		-6.3853890e-01f,
	// 		0.0f,
	// 		-6.8683788e-05f,
	// 		6.1795981e-05f,
	// 		-2.9789597e-01f,
	// 		0.0f,
	// 		0.0f
	// 	};
	// 	memcpy(
	// 		DenseLayer->Bias.Data,
	// 		FirstLayerBias,
	// 		sizeof(FirstLayerBias)
	// 	);

	// 	dense_layer* SecondDenseLayer = (dense_layer*) (
	// 		NeuralNet->FirstLink->Next->Next->Data
	// 	);
	// 	float SecondLayerWeights[] = {
	// 		0.7474555f,
	// 		-1.215051f,
	// 		-0.55316067f,
	// 		0.9348931f,
	// 		0.5940272f,
	// 		-0.53985476f,
	// 		-0.42657337f,
	// 		-0.5814253f
	// 	};
	// 	memcpy(
	// 		SecondDenseLayer->Weights.Data,
	// 		SecondLayerWeights,
	// 		sizeof(SecondLayerWeights)
	// 	);
	// 	float SecondLayerBias[] = {0.0f};
	// 	memcpy(
	// 		SecondDenseLayer->Bias.Data,
	// 		SecondLayerBias,
	// 		sizeof(SecondLayerBias)
	// 	);
		
	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"GoodXorForward",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: XOR Forward

	// // SECTION START: forward XOR with good initial weights
	// {
	// 	// NOTE: in keras, 8 neurons and one dense layer + RELU seems 
	// 	// CONT: to work. no momentum needed for high-dimensional stuff 
	// 	// CONT: 2000 epochs were needed too
	// 	uint32_t BatchSize = 4;
	// 	uint32_t InputDim = 2;
	// 	uint32_t HiddenDim = 8;
	// 	uint32_t OutputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	SetMatrixElement(Inputs, 0, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 0, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 1, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 1, 1, 1.0f);

	// 	SetMatrixElement(Inputs, 2, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 2, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 3, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 3, 1, 1.0f);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		4
	// 	);
	// 	AddDense(NeuralNet, HiddenDim);
	// 	AddRelu(NeuralNet);
	// 	AddDense(NeuralNet, OutputDim);

	// 	matrix* Labels;
	// 	AllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	// NOTE: this is set up to converge to a xor function
	// 	SetMatrixElement(Labels, 0, 0, 0.0f);
	// 	SetMatrixElement(Labels, 1, 0, 1.0f);
	// 	SetMatrixElement(Labels, 2, 0, 1.0f);
	// 	SetMatrixElement(Labels, 3, 0, 0.0f);

	// 	neural_net_trainer* Trainer;
	// 	AllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.1f,
	// 		LayerType_Mse
	// 	);

	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	float FirstLayerWeights[] = {
	// 		-0.6014779f,
	// 		0.6651714f,
	// 		-0.33612493f,
	// 		0.7284934f,
	// 		0.49762666f,
	// 		-0.33008203f,
	// 		-0.66281337f,
	// 		-0.7124146f,
	// 		0.6431314f,
	// 		0.6383204f,
	// 		-0.44230828f,
	// 		-0.7284539f,
	// 		0.7563155f,
	// 		0.29757506f,
	// 		-0.2068302f,
	// 		-0.04522699f
	// 	};
	// 	memcpy(
	// 		DenseLayer->Weights.Data,
	// 		FirstLayerWeights,
	// 		sizeof(FirstLayerWeights)
	// 	);
	// 	float FirstLayerBias[] = {
	// 		-3.8563503e-06f,
	// 		-6.3853890e-01f,
	// 		0.0f,
	// 		-6.8683788e-05f,
	// 		6.1795981e-05f,
	// 		-2.9789597e-01f,
	// 		0.0f,
	// 		0.0f
	// 	};
	// 	memcpy(
	// 		DenseLayer->Bias.Data,
	// 		FirstLayerBias,
	// 		sizeof(FirstLayerBias)
	// 	);

	// 	dense_layer* SecondDenseLayer = (dense_layer*) (
	// 		NeuralNet->FirstLink->Next->Next->Data
	// 	);
	// 	float SecondLayerWeights[] = {
	// 		0.7474555f,
	// 		-1.215051f,
	// 		-0.55316067f,
	// 		0.9348931f,
	// 		0.5940272f,
	// 		-0.53985476f,
	// 		-0.42657337f,
	// 		-0.5814253f
	// 	};
	// 	memcpy(
	// 		SecondDenseLayer->Weights.Data,
	// 		SecondLayerWeights,
	// 		sizeof(SecondLayerWeights)
	// 	);
	// 	float SecondLayerBias[] = {0.0f};
	// 	memcpy(
	// 		SecondDenseLayer->Bias.Data,
	// 		SecondLayerBias,
	// 		sizeof(SecondLayerBias)
	// 	);

	// 	TrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100,
	// 		false
	// 	);

	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"ForwardXor_StaticTraining",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: forward XOR with good initial weights

	// // SECTION START: forward XOR with close to perfect initial weights
	// {
	// 	// NOTE: in keras, 8 neurons and one dense layer + RELU seems 
	// 	// CONT: to work. no momentum needed for high-dimensional stuff 
	// 	// CONT: 2000 epochs were needed too
	// 	uint32_t BatchSize = 4;
	// 	uint32_t InputDim = 2;
	// 	uint32_t HiddenDim = 8;
	// 	uint32_t OutputDim = 1;

	// 	matrix* Inputs;
	// 	AllocMatrix(&Inputs, BatchSize, InputDim);
	// 	SetMatrixElement(Inputs, 0, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 0, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 1, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 1, 1, 1.0f);

	// 	SetMatrixElement(Inputs, 2, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 2, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 3, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 3, 1, 1.0f);

	// 	neural_net* NeuralNet = NULL;
	// 	AllocNeuralNet(
	// 		&NeuralNet,
	// 		BatchSize,
	// 		InputDim,
	// 		4
	// 	);
	// 	AddDense(NeuralNet, HiddenDim);
	// 	AddRelu(NeuralNet);
	// 	AddDense(NeuralNet, OutputDim);

	// 	matrix* Labels;
	// 	AllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	// NOTE: this is set up to converge to a xor function
	// 	SetMatrixElement(Labels, 0, 0, 0.0f);
	// 	SetMatrixElement(Labels, 1, 0, 1.0f);
	// 	SetMatrixElement(Labels, 2, 0, 1.0f);
	// 	SetMatrixElement(Labels, 3, 0, 0.0f);

	// 	neural_net_trainer* Trainer;
	// 	AllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.1f,
	// 		LayerType_Mse
	// 	);

	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	float FirstLayerWeights[] = {
	// 		-0.6f,
	// 		0.7f,
	// 		-0.3f,
	// 		0.7f,
	// 		0.5f,
	// 		-0.3f,
	// 		-0.7f,
	// 		-0.7f,
	// 		0.6f,
	// 		0.6f,
	// 		-0.4f,
	// 		-0.7f,
	// 		0.8f,
	// 		0.3f,
	// 		-0.2f,
	// 		0.0f
	// 	};
	// 	memcpy(
	// 		DenseLayer->Weights.Data,
	// 		FirstLayerWeights,
	// 		sizeof(FirstLayerWeights)
	// 	);
	// 	float FirstLayerBias[] = {
	// 		-4e-06f,
	// 		-6e-01f,
	// 		0.0f,
	// 		-7e-05f,
	// 		6e-05f,
	// 		-3e-01f,
	// 		0.0f,
	// 		0.0f
	// 	};
	// 	memcpy(
	// 		DenseLayer->Bias.Data,
	// 		FirstLayerBias,
	// 		sizeof(FirstLayerBias)
	// 	);

	// 	dense_layer* SecondDenseLayer = (dense_layer*) (
	// 		NeuralNet->FirstLink->Next->Next->Data
	// 	);
	// 	float SecondLayerWeights[] = {
	// 		0.7f,
	// 		-1.0f,
	// 		-0.6f,
	// 		0.9f,
	// 		0.6f,
	// 		-0.5f,
	// 		-0.4f,
	// 		-0.6f
	// 	};
	// 	memcpy(
	// 		SecondDenseLayer->Weights.Data,
	// 		SecondLayerWeights,
	// 		sizeof(SecondLayerWeights)
	// 	);
	// 	float SecondLayerBias[] = {0.0f};
	// 	memcpy(
	// 		SecondDenseLayer->Bias.Data,
	// 		SecondLayerBias,
	// 		sizeof(SecondLayerBias)
	// 	);

	// 	TrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100,
	// 		false
	// 	);

	// 	matrix* Predictions = NULL;
	// 	NeuralNetForward(
	// 		NeuralNet,
	// 		Inputs,
	// 		NULL,
	// 		&Predictions,
	// 		NULL
	// 	);

	// 	TestMatrixResult(
	// 		Predictions,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"ForwardXor_Convergence",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: forward XOR with close to perfect initial weights

	// // SECTION START: MNIST with MSE
	// printf("Starting MNIST training. This may take a 2-4 minutes...\n");
	// {
	// 	uint32_t MiniBatchSize = 32;
	// 	uint32_t TrainingSamples = 2048;
	// 	uint32_t TestSamples = 100;
	// 	uint32_t Epochs = 30;
	// 	float TrainingAccuracyThreshold = 0.99f;
	// 	float LossThreshold = -0.00001f;
	// 	float LearningRate = 0.1f;
	// 	bool PrintTraining = true;

	// 	snprintf(
	// 		FilePathBuffer,
	// 		sizeof(FilePathBuffer),
	// 		"%s/%s",
	// 		TestDataDirectory,
	// 		"mnist_train.csv"
	// 	);

	// 	matrix* Data;
	// 	matrix* Labels;
	// 	AllocMatrix(&Data, TrainingSamples, MNIST_DATA_SIZE);
	// 	MatrixClear(Data);
	// 	AllocMatrix(&Labels, TrainingSamples, MNIST_CLASS_COUNT);
	// 	MatrixClear(Labels);
	// 	int Result = LoadMnistDigitCsv(
	// 		Data, Labels, TrainingSamples, FilePathBuffer
	// 	);

	// 	if(Result == 0)
	// 	{
	// 		neural_net* NeuralNet = NULL;
	// 		AllocNeuralNet(
	// 			&NeuralNet,
	// 			MiniBatchSize,
	// 			MNIST_DATA_SIZE,
	// 			4
	// 		);
	// 		uint32_t HiddenDim = 64;
	// 		AddDense(NeuralNet, HiddenDim);
	// 		AddRelu(NeuralNet);
	// 		AddDense(NeuralNet, HiddenDim);
	// 		AddRelu(NeuralNet);
	// 		AddDense(NeuralNet, MNIST_CLASS_COUNT);

	// 		neural_net_trainer* Trainer;
	// 		AllocNeuralNetTrainer(
	// 			&Trainer,
	// 			NeuralNet,
	// 			LearningRate,
	// 			LayerType_Mse,
	// 			MiniBatchSize,
	// 			Labels->NumColumns
	// 		);

	// 		neural_net* FullBatchNnViewer = NULL;
	// 		ResizedNeuralNet(&FullBatchNnViewer, NeuralNet, TrainingSamples);
	// 		neural_net* TestNnViewer = NULL;
	// 		ResizedNeuralNet(&TestNnViewer, NeuralNet, TestSamples);

	// 		int64_t StartClock = Win32GetWallClock();
	// 		TrainNeuralNetMiniBatch(
	// 			Trainer,
	// 			NeuralNet,
	// 			Data,
	// 			Labels,
	// 			Epochs,
	// 			true,
	// 			PrintTraining,
	// 			TrainingAccuracyThreshold,
	// 			LossThreshold,
	// 			FullBatchNnViewer
	// 		);
	// 		int64_t EndClock = Win32GetWallClock(); 
	// 		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 		printf("Training time seconds: %f\n", Seconds);

	// 		float TrainingAccuracy = TopOneAccuracy(
	// 			FullBatchNnViewer, Data, Labels
	// 		);
	// 		printf("TrainingAccuracy = %f\n", TrainingAccuracy);

	// 		snprintf(
	// 			FilePathBuffer,
	// 			sizeof(FilePathBuffer),
	// 			"%s/%s",
	// 			TestDataDirectory,
	// 			"mnist_test.csv"
	// 		);

	// 		matrix* TestData;
	// 		matrix* TestLabels;
	// 		AllocMatrix(&TestData, TestSamples, MNIST_DATA_SIZE);
	// 		MatrixClear(TestData);
	// 		AllocMatrix(&TestLabels, TestSamples, MNIST_CLASS_COUNT);
	// 		MatrixClear(TestLabels);
	// 		Result = LoadMnistDigitCsv(
	// 			TestData, TestLabels, TestSamples, FilePathBuffer
	// 		);
	// 		float TestAccuracy = TopOneAccuracy(
	// 			TestNnViewer, TestData, TestLabels
	// 		);
	// 		printf("TestAccuracy = %f\n", TestAccuracy);

	// 		if(TestAccuracy < 0.9f)
	// 		{
	// 			printf("MNIST training test failed\n");
	// 		}

	// 		// SECTION START: test model saving and loading
	// 		snprintf(
	// 			FilePathBuffer,
	// 			sizeof(FilePathBuffer),
	// 			"%s/%s",
	// 			TestDataDirectory,
	// 			"models"
	// 		);
	// 		if(!PathFileExistsA(FilePathBuffer))
	// 		{
	// 			CreateDirectoryA(
	// 				FilePathBuffer,
	// 				NULL
	// 			);
	// 		}
	// 		snprintf(
	// 			FilePathBuffer,
	// 			sizeof(FilePathBuffer),
	// 			"%s/models/mnist_%dsamples.model",
	// 			TestDataDirectory,
	// 			TrainingSamples
	// 		);
	// 		SaveNeuralNet(NeuralNet, FilePathBuffer);

	// 		neural_net* LoadedNeuralNet;
	// 		LoadNeuralNet(
	// 			&LoadedNeuralNet, FilePathBuffer, TestSamples, 4
	// 		);

	// 		float LoadedNnTestAccuracy = TopOneAccuracy(
	// 			LoadedNeuralNet, TestData, TestLabels
	// 		);
	// 		printf("Loaded NN TestAccuracy = %f\n", LoadedNnTestAccuracy);

	// 		// SECTION STOP: test model saving and loading

	// 		// SECTION START: test freeing neural nets
	// 		// TODO: add a check for available memory before and after
	// 		FreeNeuralNetTrainer(Trainer);
	// 		FreeNeuralNet(NeuralNet);
	// 		FreeNeuralNet(LoadedNeuralNet);
	// 		FreeResizedNeuralNet(FullBatchNnViewer);
	// 		FreeResizedNeuralNet(TestNnViewer);
	// 		// SECTION STOP: test freeing neural nets
	// 	}
	// 	else
	// 	{
	// 		printf("Unable to run mnist test\n");
	// 	}
	// }
	// // SECTION STOP: MNIST with MSE

	// // SECTION START: MNIST with Softmax + X-entropy
	// printf("Starting MNIST training with softmax and X-entropy. This may take a 2-4 minutes...\n");
	// {
	// 	uint32_t MiniBatchSize = 32;
	// 	uint32_t TrainingSamples = 2048;
	// 	uint32_t TestSamples = 100;
	// 	uint32_t Epochs = 30;
	// 	float TrainingAccuracyThreshold = 0.99f;
	// 	float LossThreshold = -0.00001f;
	// 	float LearningRate = 0.1f;
	// 	bool PrintTraining = true;

	// 	snprintf(
	// 		FilePathBuffer,
	// 		sizeof(FilePathBuffer),
	// 		"%s/%s",
	// 		TestDataDirectory,
	// 		"mnist_train.csv"
	// 	);

	// 	matrix* Data;
	// 	matrix* Labels;
	// 	AllocMatrix(&Data, TrainingSamples, MNIST_DATA_SIZE);
	// 	MatrixClear(Data);
	// 	AllocMatrix(&Labels, TrainingSamples, MNIST_CLASS_COUNT);
	// 	MatrixClear(Labels);
	// 	int Result = LoadMnistDigitCsv(
	// 		Data, Labels, TrainingSamples, FilePathBuffer
	// 	);

	// 	if(Result == 0)
	// 	{
	// 		neural_net* NeuralNet = NULL;
	// 		AllocNeuralNet(
	// 			&NeuralNet,
	// 			MiniBatchSize,
	// 			MNIST_DATA_SIZE,
	// 			4
	// 		);
	// 		uint32_t HiddenDim = 64;
	// 		AddDense(NeuralNet, HiddenDim);
	// 		AddRelu(NeuralNet);
	// 		AddDense(NeuralNet, HiddenDim);
	// 		AddRelu(NeuralNet);
	// 		AddDense(NeuralNet, MNIST_CLASS_COUNT);
	// 		AddSoftmaxCrossEntropy(NeuralNet);
	// 		neural_net_trainer* Trainer;
	// 		AllocNeuralNetTrainer(
	// 			&Trainer,
	// 			NeuralNet,
	// 			LearningRate,
	// 			LayerType_Count,
	// 			MiniBatchSize,
	// 			Labels->NumColumns
	// 		);

	// 		neural_net* FullBatchNnViewer = NULL;
	// 		ResizedNeuralNet(&FullBatchNnViewer, NeuralNet, TrainingSamples);
	// 		neural_net* TestNnViewer = NULL;
	// 		ResizedNeuralNet(&TestNnViewer, NeuralNet, TestSamples);

	// 		int64_t StartClock = Win32GetWallClock();
	// 		TrainNeuralNetMiniBatch(
	// 			Trainer,
	// 			NeuralNet,
	// 			Data,
	// 			Labels,
	// 			Epochs,
	// 			true,
	// 			PrintTraining,
	// 			TrainingAccuracyThreshold,
	// 			LossThreshold,
	// 			FullBatchNnViewer
	// 		);
	// 		int64_t EndClock = Win32GetWallClock(); 
	// 		float Seconds = Win32GetSecondsElapsed(StartClock, EndClock);
	// 		printf("Training time seconds: %f\n", Seconds);

	// 		float TrainingAccuracy = TopOneAccuracy(
	// 			FullBatchNnViewer, Data, Labels
	// 		);
	// 		printf("TrainingAccuracy = %f\n", TrainingAccuracy);

	// 		snprintf(
	// 			FilePathBuffer,
	// 			sizeof(FilePathBuffer),
	// 			"%s/%s",
	// 			TestDataDirectory,
	// 			"mnist_test.csv"
	// 		);

	// 		matrix* TestData;
	// 		matrix* TestLabels;
	// 		AllocMatrix(&TestData, TestSamples, MNIST_DATA_SIZE);
	// 		MatrixClear(TestData);
	// 		AllocMatrix(&TestLabels, TestSamples, MNIST_CLASS_COUNT);
	// 		MatrixClear(TestLabels);
	// 		Result = LoadMnistDigitCsv(
	// 			TestData, TestLabels, TestSamples, FilePathBuffer
	// 		);
	// 		float TestAccuracy = TopOneAccuracy(
	// 			TestNnViewer, TestData, TestLabels
	// 		);
	// 		printf("TestAccuracy = %f\n", TestAccuracy);

	// 		if(TestAccuracy < 0.9f)
	// 		{
	// 			printf("MNIST training test failed\n");
	// 		}

	// 		// // SECTION START: test model saving and loading
	// 		// snprintf(
	// 		// 	FilePathBuffer,
	// 		// 	sizeof(FilePathBuffer),
	// 		// 	"%s/%s",
	// 		// 	TestDataDirectory,
	// 		// 	"models"
	// 		// );
	// 		// if(!PathFileExistsA(FilePathBuffer))
	// 		// {
	// 		// 	CreateDirectoryA(
	// 		// 		FilePathBuffer,
	// 		// 		NULL
	// 		// 	);
	// 		// }
	// 		// snprintf(
	// 		// 	FilePathBuffer,
	// 		// 	sizeof(FilePathBuffer),
	// 		// 	"%s/models/mnist_%dsamples.model",
	// 		// 	TestDataDirectory,
	// 		// 	TrainingSamples
	// 		// );
	// 		// SaveNeuralNet(NeuralNet, FilePathBuffer);

	// 		// neural_net* LoadedNeuralNet;
	// 		// LoadNeuralNet(
	// 		// 	&LoadedNeuralNet, FilePathBuffer, TestSamples, 4
	// 		// );

	// 		// float LoadedNnTestAccuracy = TopOneAccuracy(
	// 		// 	LoadedNeuralNet, TestData, TestLabels
	// 		// );
	// 		// printf("Loaded NN TestAccuracy = %f\n", LoadedNnTestAccuracy);

	// 		// // SECTION STOP: test model saving and loading

	// 		// // SECTION START: test freeing neural nets
	// 		// // TODO: add a check for available memory before and after
	// 		// FreeNeuralNetTrainer(Trainer);
	// 		// FreeNeuralNet(NeuralNet);
	// 		// FreeNeuralNet(LoadedNeuralNet);
	// 		// FreeResizedNeuralNet(FullBatchNnViewer);
	// 		// FreeResizedNeuralNet(TestNnViewer);
	// 		// // SECTION STOP: test freeing neural nets
	// 	}
	// 	else
	// 	{
	// 		printf("Unable to run mnist test\n");
	// 	}
	// }
	// // SECTION STOP: MNIST with Softmax + X Entropy
}