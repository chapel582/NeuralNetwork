#include "neural_net.h"

#include "matrix.h"
#include "matrix.cpp"

#include "matrix_test.cpp"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

__device__
float CudaGetMatrixElement(matrix* Matrix, uint32_t Row, uint32_t Column)
{
	assert(Row < Matrix->NumRows);
	assert(Column < Matrix->NumColumns);
	float* Element = Matrix->Data + Row * Matrix->NumColumns + Column;
	return *Element;
}

__device__
float CudaGetMatrixElement(matrix* Matrix, uint32_t ElementIndex)
{
	// NOTE: made available if the Row, Column asserts in the standard 
	// CONT: GetMatrixElement isn't needed. Mostly used for when you don't care
	// CONT: if you have a row or column matrix
	assert(ElementIndex < (Matrix->NumRows * Matrix->NumColumns));
	float* Element = Matrix->Data + ElementIndex;
	return *Element;
}

__device__
void CudaSetMatrixElement(
	matrix* Matrix, uint32_t Row, uint32_t Column, float Value
)
{
	assert(Row < Matrix->NumRows);
	assert(Column < Matrix->NumColumns);
	float* Element = Matrix->Data + Row * Matrix->NumColumns + Column;
	*Element = Value;
}

void CudaInitMatrix(matrix* Matrix, uint32_t NumRows, uint32_t NumColumns)
{
	*Matrix = {};
	Matrix->NumRows = NumRows;
	Matrix->NumColumns = NumColumns;
	cudaMallocManaged(&Matrix->Data, GetMatrixDataSize(Matrix));
	memset(Matrix->Data, 0, GetMatrixDataSize(Matrix));
}

void CudaAllocMatrix(matrix** Result, uint32_t NumRows, uint32_t NumColumns)
{
	cudaMallocManaged(Result, sizeof(matrix));
	matrix* Matrix = *Result;
	CudaInitMatrix(Matrix, NumRows, NumColumns);
}

void CudaAllocMultResultMatrix(matrix** Result, matrix* M1, matrix* M2)
{
	// NOTE: allocates a matrix that would result from the matrix multiplication
	// CONT: of M1 and M2
	CudaAllocMatrix(Result, M1->NumRows, M2->NumColumns);
}

void CudaAllocM1TransposeMultResultMatrix(
	matrix** Result, matrix* M1, matrix* M2
)
{
	// NOTE: allocates a matrix that would result from the matrix multiplication
	// CONT: of M1 tranposed and M2
	CudaAllocMatrix(Result, M1->NumColumns, M2->NumColumns);
}

void CudaAllocM2TransposeMultResultMatrix(
	matrix** Result, matrix* M1, matrix* M2
)
{
	// NOTE: allocates a matrix that would result from the matrix multiplication
	// CONT: of M1 tranposed and M2
	CudaAllocMatrix(Result, M1->NumRows, M2->NumRows);
}

void CudaAllocM1M2TransposeMultResultMatrix(
	matrix** Result, matrix* M1, matrix* M2
)
{
	// NOTE: allocates a matrix that would result from the matrix multiplication
	// CONT: of M1 tranposed and M2
	CudaAllocMatrix(Result, M1->NumColumns, M2->NumRows);
}

void CudaAllocMatrixMeanResult(matrix** Result, matrix* M1)
{
	CudaAllocMatrix(Result, 1, M1->NumColumns);
}

// TODO: get cuda memory free in here
// void FreeMatrixData(matrix Matrix)
// {
// 	free(Matrix.Data);
// }

// void FreeMatrix(matrix* Matrix)
// {
// 	FreeMatrixData(*Matrix);
// 	free(Matrix);
// }

inline int GetNumBlocks(int Range, int BlockSize)
{
	return (Range + BlockSize - 1) / BlockSize;
}

__global__
void CudaMatrixMultCore(matrix* M1, matrix* M2, matrix* Result)
{	
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	for(uint32_t Row = Start; Row < M1->NumRows; Row += Stride)
	{
		for(uint32_t Column = 0; Column < M2->NumColumns; Column++)
		{
			float DotProduct = 0.0f;
			for(uint32_t DPIndex = 0; DPIndex < M1->NumColumns; DPIndex++)
			{
				DotProduct += (
					CudaGetMatrixElement(M1, Row, DPIndex) * 
					CudaGetMatrixElement(M2, DPIndex, Column)
				);
			}
			CudaSetMatrixElement(Result, Row, Column, DotProduct);
		}
	}
}

void CudaMatrixMult(matrix* M1, matrix* M2, matrix* Result)
{
	assert(M1->NumRows == M2->NumColumns);
	// NOTE: not sure if this should be a variable or queried or tracked with 
	// CONT: a data structure
	int BlockSize = 256;

	// NOTE: NumBlocks is always at least one, and grows as the data to 
	// NOTE: process grows
	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize);
	CudaMatrixMultCore<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__global__
void CudaAddVectorToRowsCore(matrix* M1, matrix* Vector, matrix* Result)
{
	/*NOTE:
	Because the vector is one-dimensional, it doesn't matter whether you pass 
	Col into the row or the column 
	a nice consequence of this is that it doesn't matter whether you pass in a 
	row vector or a column vector. It will project nicely as long as the non-one
	dimension is equal to the number of columns of M1
	*/

	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	for(uint32_t Row = Start; Row < M1->NumRows; Row += Stride)
	{
		for(uint32_t Col = 0; Col < M1->NumColumns; Col++)
		{
			CudaSetMatrixElement(
				Result,
				Row,
				Col,
				(
					CudaGetMatrixElement(M1, Row, Col) + 
					CudaGetMatrixElement(Vector, Col)
				)
			);
		}
	}
}

void CudaAddVectorToRows(matrix* M1, matrix* Vector, matrix* Result)
{
	// NOTE: this function is equivalent to adding two matrices, M1 and M2,
	// CONT: where M2 has the same values in each row (Vector) 
	// NOTE: there's no reason to allocate a huge matrix just for this, so this 
	// CONT: method is used instead
	assert(
		(M1->NumColumns == Vector->NumColumns) ||
		(M1->NumColumns == Vector->NumRows)
	);

	// NOTE: not sure if this should be a variable or queried or tracked with 
	// CONT: a data structure
	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize);
	CudaAddVectorToRowsCore<<<NumBlocks, BlockSize>>>(M1, Vector, Result);
	cudaDeviceSynchronize();
}

__global__
void CudaMatrixAddCore(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	for(uint32_t Row = Start; Row < M1->NumRows; Row += Stride)
	{
		for(uint32_t Col = 0; Col < M1->NumColumns; Col++)
		{
			CudaSetMatrixElement(
				Result,
				Row,
				Col,
				CudaGetMatrixElement(M1, Row, Col) + 
				CudaGetMatrixElement(M2, Row, Col)
			);
		}
	}
}

void CudaMatrixAdd(matrix* M1, matrix* M2, matrix* Result)
{
	assert(M1->NumRows == M2->NumRows);
	assert(M1->NumColumns == M2->NumColumns);
	
	// NOTE: not sure if this should be a variable or queried or tracked with 
	// CONT: a data structure
	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize);
	CudaMatrixAddCore<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__global__
void CudaMatrixMultM1TransposeCore(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	for(uint32_t Row = Start; Row < M1->NumColumns; Row += Stride)
	{
		for(uint32_t Column = 0; Column < M2->NumColumns; Column++)
		{
			float DotProduct = 0.0f;
			for(uint32_t DPIndex = 0; DPIndex < M1->NumRows; DPIndex++)
			{
				DotProduct += (
					CudaGetMatrixElement(M1, DPIndex, Row) * 
					CudaGetMatrixElement(M2, DPIndex, Column)
				);
			}
			CudaSetMatrixElement(Result, Row, Column, DotProduct);
		}
	}
}

void CudaMatrixMultM1Transpose(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: For transpose multiplication without allocating and initializing
	// CONT: a new matrix
	// NOTE: the number of rows in M1 should equal the number of rows in M2

	assert(M1->NumRows == M2->NumRows);

	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumColumns, BlockSize);
	CudaMatrixMultM1TransposeCore<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__global__
void CudaMatrixMultM2TransposeCore(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	for(uint32_t Row = Start; Row < M1->NumRows; Row += Stride)
	{
		for(uint32_t Column = 0; Column < M2->NumRows; Column++)
		{
			float DotProduct = 0.0f;
			for(uint32_t DPIndex = 0; DPIndex < M1->NumColumns; DPIndex++)
			{
				DotProduct += (
					CudaGetMatrixElement(M1, Row, DPIndex) * 
					CudaGetMatrixElement(M2, Column, DPIndex)
				);
			}
			CudaSetMatrixElement(Result, Row, Column, DotProduct);
		}
	}
}

void CudaMatrixMultM2Transpose(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: For transpose multiplication without allocating and initializing
	// CONT: a new matrix
	// NOTE: the number of columns in M1 should equal the number of columns in M2

	assert(M1->NumColumns == M2->NumColumns);

	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize);
	CudaMatrixMultM2TransposeCore<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__global__
void CudaMatrixMultM1M2TransposeCore(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	for(uint32_t Row = Start; Row < M1->NumColumns; Row += Stride)
	{
		for(uint32_t Column = 0; Column < M2->NumRows; Column++)
		{
			float DotProduct = 0.0f;
			for(uint32_t DPIndex = 0; DPIndex < M1->NumRows; DPIndex++)
			{
				DotProduct += (
					CudaGetMatrixElement(M1, DPIndex, Row) * 
					CudaGetMatrixElement(M2, Column, DPIndex)
				);
			}
			CudaSetMatrixElement(Result, Row, Column, DotProduct);
		}
	}
}

void CudaMatrixMultM1M2Transpose(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: For transpose multiplication without allocating and initializing
	// CONT: a new matrix
	assert(M1->NumRows == M2->NumColumns);

	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumColumns, BlockSize);
	CudaMatrixMultM1M2TransposeCore<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__global__
void CudaMatrixScalarMultCore(float Scalar, matrix* M1, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	for(uint32_t Row = Start; Row < M1->NumRows; Row += Stride)
	{
		for(uint32_t Column = 0; Column < M1->NumColumns; Column++)
		{
			float NewValue = Scalar * CudaGetMatrixElement(M1, Row, Column);
			CudaSetMatrixElement(Result, Row, Column, NewValue);
		}
	}
}

void CudaMatrixScalarMult(float Scalar, matrix* M1, matrix* Result)
{
	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize);
	CudaMatrixScalarMultCore<<<NumBlocks, BlockSize>>>(Scalar, M1, Result);
	cudaDeviceSynchronize();
}

__global__
void CudaMatrixSubtractCore(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	for(uint32_t Row = Start; Row < M1->NumRows; Row += Stride)
	{
		for(uint32_t Col = 0; Col < M1->NumColumns; Col++)
		{
			CudaSetMatrixElement(
				Result,
				Row,
				Col,
				CudaGetMatrixElement(M1, Row, Col) - 
				CudaGetMatrixElement(M2, Row, Col)
			);
		}
	}
}

void CudaMatrixSubtract(matrix* M1, matrix* M2, matrix* Result)
{
	assert(M1->NumRows == M2->NumRows);
	assert(M1->NumColumns == M2->NumColumns);

	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize);

	CudaMatrixSubtractCore<<<BlockSize, NumBlocks>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaMatrixScalarMultCoreColStride(
	float Scalar, matrix* M1, matrix* Result, int Start, int Stride
)
{
	// NOTE: the number of columns in M1 should equal the number of rows in M2
	// NOTE: mostly a helper function for the mean function
	for(uint32_t Row = 0; Row < M1->NumRows; Row++)
	{
		for(uint32_t Column = Start; Column < M1->NumColumns; Column += Stride)
		{
			float NewValue = Scalar * CudaGetMatrixElement(M1, Row, Column);
			CudaSetMatrixElement(Result, Row, Column, NewValue);
		}
	}
}

__global__
void CudaMatrixMeanCore(matrix* M1, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	CudaMatrixScalarMultCoreColStride(0.0f, Result, Result, Start, Stride);
	for(uint32_t Row = 0; Row < M1->NumRows; Row++)
	{
		for(uint32_t Col = Start; Col < M1->NumColumns; Col += Stride)
		{
			float NewValue = (
				CudaGetMatrixElement(Result, 0, Col) + 
				CudaGetMatrixElement(M1, Row, Col)
			);
			CudaSetMatrixElement(Result, 0, Col, NewValue);
		}
	}
	CudaMatrixScalarMultCoreColStride(
		1.0f / M1->NumRows, Result, Result, Start, Stride
	);
}

void CudaMatrixMean(matrix* M1, matrix* Result)
{
	/*NOTE:
	This function finds the sum of all the row vectors of matrix M1 and divides
	that sum by the number of rows. 

	M1 Dimensions: N x M
	Result Dimensions: 1 x M
	*/
	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumColumns, BlockSize);
	CudaMatrixMeanCore<<<NumBlocks, BlockSize>>>(M1, Result);
	cudaDeviceSynchronize();
}

void CudaAllocDenseLayer(
	dense_layer** Result, uint32_t InputDim, uint32_t OutputDim
)
{
	cudaMallocManaged(Result, sizeof(dense_layer));
	dense_layer* DenseLayer = *Result;
	*DenseLayer = {};
	CudaInitMatrix(&DenseLayer->Weights, InputDim, OutputDim);
	CudaInitMatrix(&DenseLayer->Bias, 1, OutputDim);
}

void CudaDenseForward(matrix* Inputs, dense_layer* DenseLayer, matrix* Results)
{
	CudaMatrixMult(Inputs, &DenseLayer->Weights, Results);
	CudaAddVectorToRows(Results, &DenseLayer->Bias, Results);	
}

void CudaAllocDenseLayerTrain(
	dense_layer_train_data** Result,
	dense_layer* DenseLayer,
	float LearningRate,
	uint32_t BatchSize
)
{
	cudaMallocManaged(Result, sizeof(dense_layer_train_data));
	dense_layer_train_data* TrainData = *Result;
	*TrainData = {};
	TrainData->LearningRate = LearningRate; 
	CudaInitMatrix(
		&TrainData->WeightsDelta,
		DenseLayer->Weights.NumRows,
		DenseLayer->Weights.NumColumns
	);
	CudaInitMatrix(
		&TrainData->BiasDelta,
		DenseLayer->Bias.NumRows,
		DenseLayer->Bias.NumColumns
	);
	CudaInitMatrix(
		&TrainData->LayerGradient, BatchSize, DenseLayer->Weights.NumRows
	);
}

void CudaDenseBack(
	matrix* Inputs,
	matrix* NextLayerGradient,
	dense_layer* DenseLayer,
	dense_layer_train_data* TrainData
)
{
	CudaMatrixMultM2Transpose(
		NextLayerGradient, &DenseLayer->Weights, &TrainData->LayerGradient
	);

	CudaMatrixMultM1Transpose(
		Inputs, NextLayerGradient, &TrainData->WeightsDelta
	);
	CudaMatrixScalarMult(
		TrainData->LearningRate,
		&TrainData->WeightsDelta,
		&TrainData->WeightsDelta
	);
	CudaMatrixAdd(
		&DenseLayer->Weights,
		&TrainData->WeightsDelta,
		&DenseLayer->Weights
	);
	
	CudaMatrixMean(NextLayerGradient, &TrainData->BiasDelta);
	CudaMatrixScalarMult(
		TrainData->LearningRate,
		&TrainData->BiasDelta,
		&TrainData->BiasDelta
	);
	CudaMatrixAdd(
		&DenseLayer->Bias,
		&TrainData->BiasDelta,
		&DenseLayer->Bias
	);
}

#define SAVE_RESULTS 0
matrix* TestMatrixResult(
	matrix* M1,
	char* FilePathBuffer,
	size_t FilePathBufferSize,
	char* TestDataDirectory,
	char* TestName,
	char* EndianString
)
{
	// NOTE: if this function is changed in both test programs 3 more times, 
	// CONT: it's time to refactor it

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
	CudaAllocMatrix(&CompareTo, M1->NumRows, M1->NumColumns);
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

int main(int argc, char* argv[])
{
	// TODO: move test code out to other file

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

	bool BigEndian = IsBigEndian();
	char EndianString[260];
	if(BigEndian)
	{
		strcpy_s(EndianString, sizeof(EndianString), "BigEndian");
	}
	else
	{
		strcpy_s(EndianString, sizeof(EndianString), "LittleEndian");
	}
	char FilePathBuffer[260];
	char FileName[260];

	// SECTION START: Matrix tests
	{
		matrix* M1;
		uint32_t NumRows = 3;
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
		CudaMatrixMult(M1, M2, MultResult);
		// NOTE: TestMatrixResult returns a matrix pointer that can be freed
		strcpy_s(FileName, sizeof(FileName), "CudaMultResult");
		TestMatrixResult(
			MultResult,
			FilePathBuffer,
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		MatrixClear(MultResult);

		matrix* M3;
		NumRows = 3;
		NumColumns = 2;
		CudaAllocMatrix(&M3, NumRows, NumColumns);
		FillMatrixConsecutive(M3);

		matrix* M4;
		NumRows = 2;
		NumColumns = 3;
		CudaAllocMatrix(&M4, NumRows, NumColumns);
		FillMatrixConsecutive(M4);

		matrix* MultResult2;
		CudaAllocMultResultMatrix(&MultResult2, M3, M4);
		CudaMatrixMult(M3, M4, MultResult2);
		strcpy_s(FileName, sizeof(FileName), "CudaNonSquareMult");
		TestMatrixResult(
			MultResult2,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* AddResult;
		CudaAllocMatrix(&AddResult, M1->NumRows, M1->NumColumns);
		CudaMatrixAdd(M1, M2, AddResult);
		strcpy_s(FileName, sizeof(FileName), "CudaMatrixAdd");
		TestMatrixResult(
			AddResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* AddVectorResult;
		CudaAllocMatrix(&AddVectorResult, M1->NumRows, M1->NumColumns);
		matrix* Vector;
		CudaAllocMatrix(&Vector, 1, M1->NumColumns);
		FillMatrixConsecutive(Vector);
		CudaAddVectorToRows(M1, Vector, AddVectorResult);
		strcpy_s(FileName, sizeof(FileName), "CudaAddVectorToRows");
		TestMatrixResult(
			AddVectorResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* M5;
		NumRows = 2;
		NumColumns = 3;
		CudaAllocMatrix(&M5, NumRows, NumColumns);
		FillMatrixConsecutive(M5);

		matrix* M6;
		NumRows = 2;
		NumColumns = 3;
		CudaAllocMatrix(&M6, NumRows, NumColumns);
		FillMatrixConsecutive(M6);

		matrix* M5TMultResult;
		CudaAllocM1TransposeMultResultMatrix(&M5TMultResult, M5, M6);
		CudaMatrixMultM1Transpose(M5, M6, M5TMultResult);
		strcpy_s(FileName, sizeof(FileName), "CudaMatrixMultM1Transpose");
		TestMatrixResult(
			M5TMultResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		MatrixClear(M5TMultResult);
		SetMatrixElement(M6, 0, 1, 7);
		SetMatrixElement(M6, 1, 2, 13);
		
		CudaMatrixMultM1Transpose(M5, M6, M5TMultResult);
		strcpy_s(
			FileName, sizeof(FileName), "CudaNonSymmetricMatrixMultM1Transpose"
		);
		TestMatrixResult(
			M5TMultResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* M6TMultResult;
		CudaAllocM2TransposeMultResultMatrix(&M6TMultResult, M5, M6);

		CudaMatrixMultM2Transpose(M5, M6, M6TMultResult);
		strcpy_s(FileName, sizeof(FileName), "CudaMatrixMultM2Transpose");
		TestMatrixResult(
			M6TMultResult,
			FilePathBuffer,
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* M7;
		NumRows = 2;
		NumColumns = 3;
		CudaAllocMatrix(&M7, NumRows, NumColumns);
		FillMatrixConsecutive(M7);

		matrix* M8;
		NumRows = 3;
		NumColumns = 2;
		CudaAllocMatrix(&M8, NumRows, NumColumns);
		FillMatrixConsecutive(M8);

		matrix* M7TM8TMultResult;
		CudaAllocM1M2TransposeMultResultMatrix(&M7TM8TMultResult, M7, M8);
		MatrixClear(M7TM8TMultResult);
		CudaMatrixMultM1M2Transpose(M7, M8, M7TM8TMultResult);
		strcpy_s(FileName, sizeof(FileName), "CudaMatrixMultM1M2Transpose");
		TestMatrixResult(
			M7TM8TMultResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* M9;
		NumRows = 3;
		NumColumns = 4;
		CudaAllocMatrix(&M9, NumRows, NumColumns);
		FillMatrixConsecutive(M9);
		
		CudaMatrixScalarMult(0.5f, M9, M9);
		strcpy_s(FileName, sizeof(FileName), "CudaMatrixScalarMult");
		TestMatrixResult(
			M9,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* M10;
		NumRows = 4;
		NumColumns = 4;
		CudaAllocMatrix(&M10, NumRows, NumColumns);
		FillMatrixConsecutive(M10);

		matrix* M10Mean;
		CudaAllocMatrixMeanResult(&M10Mean, M10);
		MatrixClear(M10Mean);
		CudaMatrixMean(M10, M10Mean);
		strcpy_s(FileName, sizeof(FileName), "CudaMatrixRowMean");
		TestMatrixResult(
			M10Mean,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* M11;
		NumRows = 3;
		NumColumns = 4;
		CudaAllocMatrix(&M11, NumRows, NumColumns);
		FillMatrixConsecutive(M11);

		matrix* M12;
		CudaAllocMatrix(&M12, NumRows, NumColumns);
		FillMatrixConsecutive(M12);
		SetMatrixElement(M12, 0, 0, -2.0f);
		matrix* SubResult;
		CudaAllocMatrix(&SubResult, NumRows, NumColumns);
		CudaMatrixSubtract(M11, M12, SubResult);

		strcpy_s(FileName, sizeof(FileName), "CudaMatrixSub");
		TestMatrixResult(
			SubResult,
			FilePathBuffer,
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);
		// NOTE: if memory starts getting hefty, free memory here
	}
	// SECTION STOP: Matrix tests

	// SECTION START: Dense layer tests
	{
		uint32_t BatchSize = 8;
		uint32_t InputDim = 4;
		uint32_t OutputDim = 3;
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
		CudaDenseForward(Inputs, DenseLayer, Outputs);
		strcpy_s(FileName, sizeof(FileName), "CudaForwardDense");
		TestMatrixResult(
			Outputs,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);

		matrix* NextLayerGradient;
		CudaAllocMatrix(&NextLayerGradient, BatchSize, OutputDim);
		FillMatrixConsecutive(NextLayerGradient);

		dense_layer_train_data* TrainData;
		CudaAllocDenseLayerTrain(&TrainData, DenseLayer, 1.0f, BatchSize);
		CudaDenseBack(
			Inputs, NextLayerGradient, DenseLayer, TrainData
		);
		strcpy_s(FileName, sizeof(FileName), "CudaDenseWeightsAfterUpdate");
		TestMatrixResult(
			&DenseLayer->Weights,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);
		strcpy_s(FileName, sizeof(FileName), "CudaDenseBiasAfterUpdate");
		TestMatrixResult(
			&DenseLayer->Bias,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);
		strcpy_s(FileName, sizeof(FileName), "CudaDenseLayerGradient");
		TestMatrixResult(
			&TrainData->LayerGradient,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			FileName,
			EndianString
		);
	}
	// SECTION STOP: Dense layer tests
	return 0;
}