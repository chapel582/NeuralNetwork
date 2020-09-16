// TODO: handle cudaMallocManaged failures
// TODO: query max block size
#include "arg_max.h"
#include "int_shuffler.h"
#include "neural_net.h"
#include "matrix.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

__device__
uint32_t CudaGetMatrixArrayCount(matrix* Matrix)
{
	return Matrix->NumRows * Matrix->NumColumns;
}

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
	assert(ElementIndex < CudaGetMatrixArrayCount(Matrix));
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

__device__
void CudaSetMatrixElement(
	matrix* Matrix, uint32_t ElementIndex, float Value
)
{
	assert(ElementIndex < CudaGetMatrixArrayCount(Matrix));
	float* Element = Matrix->Data + ElementIndex;
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

void CudaFreeMatrixData(matrix* Matrix)
{
	cudaFree(Matrix->Data);
}

void CudaAllocMatrix(matrix** Result, uint32_t NumRows, uint32_t NumColumns)
{
	cudaMallocManaged(Result, sizeof(matrix));
	matrix* Matrix = *Result;
	CudaInitMatrix(Matrix, NumRows, NumColumns);
}

void CudaFreeMatrix(matrix* Matrix)
{
	CudaFreeMatrixData(Matrix);
	cudaFree(Matrix);
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

#define MAX_GPUS 1
int GlobalMaxBlockSizeArray[MAX_GPUS];
int GlobalMaxGridDimArray[MAX_GPUS];

void CudaInitDeviceProperties(uint32_t Device)
{
	assert(Device < MAX_GPUS);
	cudaDeviceGetAttribute(
		&GlobalMaxBlockSizeArray[Device], cudaDevAttrMaxBlockDimX, Device
	);
	cudaDeviceGetAttribute(
		&GlobalMaxGridDimArray[Device], cudaDevAttrMaxGridDimX, Device
	);
}

inline uint32_t GetBlockSize(uint32_t Device)
{
	return GlobalMaxBlockSizeArray[Device];
}

uint32_t GetNumBlocks(uint32_t Range, uint32_t BlockSize, uint32_t Device)
{
	// NOTE: for getting max blocks for operations that are parallelizable 
	// CONT: without no sync

	// NOTE: NumBlocks is always at least one, and grows as the data to 
	// CONT: process grows
	
	uint32_t NumBlocks = (Range + BlockSize - 1) / BlockSize;
	int MaxBlocks = GlobalMaxGridDimArray[Device];
	if(MaxBlocks < NumBlocks)
	{
		MaxBlocks = NumBlocks;
	}
	return NumBlocks;
}

__device__
void CudaMatrixMultCore(
	matrix* M1, matrix* M2, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t CommonDim = M1->NumColumns;
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = CudaGetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Row = ResultIndex / ResultColumns;
		uint32_t Column = ResultIndex % ResultColumns;
		float DotProduct = 0.0f;
		for(uint32_t DPIndex = 0; DPIndex < CommonDim; DPIndex++)
		{
			DotProduct += (
				CudaGetMatrixElement(M1, Row, DPIndex) * 
				CudaGetMatrixElement(M2, DPIndex, Column)
			);
		}
		CudaSetMatrixElement(Result, Row, Column, DotProduct);
	}
}

__global__
void CudaMatrixMultThread(matrix* M1, matrix* M2, matrix* Result)
{	
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaMatrixMultCore(M1, M2, Result, Start, Stride);
}

void CudaMatrixMult(matrix* M1, matrix* M2, matrix* Result)
{
	assert(M1->NumColumns == M2->NumRows);
	uint32_t Device = 0;
	int BlockSize = GetBlockSize(Device);
	int NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(Result), BlockSize, Device
	);
	CudaMatrixMultThread<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaAddVectorToRowsCore(
	matrix* M1, matrix* Vector, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = CudaGetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Column = ResultIndex % ResultColumns;
		CudaSetMatrixElement(
			Result,
			ResultIndex,
			CudaGetMatrixElement(M1, ResultIndex) + 
			CudaGetMatrixElement(Vector, Column)
		);
	}
}

__global__
void CudaAddVectorToRowsThread(matrix* M1, matrix* Vector, matrix* Result)
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
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaAddVectorToRowsCore(M1, Vector, Result, Start, Stride);
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
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(M1), BlockSize, Device
	);
	CudaAddVectorToRowsThread<<<NumBlocks, BlockSize>>>(M1, Vector, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaMatrixAddCore(
	matrix* M1, matrix* M2, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t NumResultElements = CudaGetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		CudaSetMatrixElement(
			Result,
			ResultIndex,
			CudaGetMatrixElement(M1, ResultIndex) + 
			CudaGetMatrixElement(M2, ResultIndex)
		);
	}
}

__global__
void CudaMatrixAddThread(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaMatrixAddCore(M1, M2, Result, Start, Stride);
}

void CudaMatrixAdd(matrix* M1, matrix* M2, matrix* Result)
{
	assert(M1->NumRows == M2->NumRows);
	assert(M1->NumColumns == M2->NumColumns);
	
	// NOTE: not sure if this should be a variable or queried or tracked with 
	// CONT: a data structure
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(M1), BlockSize, Device
	);
	CudaMatrixAddThread<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaMatrixMultM1TransposeCore(
	matrix* M1, matrix* M2, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t CommonDim = M1->NumRows;
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = CudaGetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Row = ResultIndex / ResultColumns;
		uint32_t Column = ResultIndex % ResultColumns;
		float DotProduct = 0.0f;
		for(uint32_t DPIndex = 0; DPIndex < CommonDim; DPIndex++)
		{
			DotProduct += (
				CudaGetMatrixElement(M1, DPIndex, Row) * 
				CudaGetMatrixElement(M2, DPIndex, Column)
			);
		}
		CudaSetMatrixElement(Result, Row, Column, DotProduct);
	}
}

__global__
void CudaMatrixMultM1TransposeThread(matrix* M1, matrix* M2, matrix* Result)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;
	
	CudaMatrixMultM1TransposeCore(M1, M2, Result, Start, Stride);
}

void CudaMatrixMultM1Transpose(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: For transpose multiplication without allocating and initializing
	// CONT: a new matrix
	// NOTE: the number of rows in M1 should equal the number of rows in M2

	assert(M1->NumRows == M2->NumRows);

	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(Result), BlockSize, Device
	);
	CudaMatrixMultM1TransposeThread<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaMatrixMultM2TransposeCore(
	matrix* M1, matrix* M2, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t CommonDim = M1->NumColumns;
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = CudaGetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Row = ResultIndex / ResultColumns;
		uint32_t Column = ResultIndex % ResultColumns;
		float DotProduct = 0.0f;
		for(uint32_t DPIndex = 0; DPIndex < CommonDim; DPIndex++)
		{
			DotProduct += (
				CudaGetMatrixElement(M1, Row, DPIndex) * 
				CudaGetMatrixElement(M2, Column, DPIndex)
			);
		}
		CudaSetMatrixElement(Result, Row, Column, DotProduct);
	}
}

__global__
void CudaMatrixMultM2TransposeThread(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaMatrixMultM2TransposeCore(M1, M2, Result, Start, Stride);
}

void CudaMatrixMultM2Transpose(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: For transpose multiplication without allocating and initializing
	// CONT: a new matrix
	// NOTE: the number of columns in M1 should equal the number of columns in M2

	assert(M1->NumColumns == M2->NumColumns);

	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(Result), BlockSize, Device
	);
	CudaMatrixMultM2TransposeThread<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaMatrixMultM1M2TransposeCore(
	matrix* M1, matrix* M2, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t CommonDim = M1->NumRows;
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = CudaGetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Row = ResultIndex / ResultColumns;
		uint32_t Column = ResultIndex % ResultColumns;
		float DotProduct = 0.0f;
		for(uint32_t DPIndex = 0; DPIndex < CommonDim; DPIndex++)
		{
			DotProduct += (
				CudaGetMatrixElement(M1, DPIndex, Row) * 
				CudaGetMatrixElement(M2, Column, DPIndex)
			);
		}
		CudaSetMatrixElement(Result, Row, Column, DotProduct);
	}
}

__global__
void CudaMatrixMultM1M2TransposeThread(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaMatrixMultM1M2TransposeCore(M1, M2, Result, Start, Stride);
}

void CudaMatrixMultM1M2Transpose(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: For transpose multiplication without allocating and initializing
	// CONT: a new matrix
	assert(M1->NumRows == M2->NumColumns);

	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(Result), BlockSize, Device
	);
	CudaMatrixMultM1M2TransposeThread<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaMatrixScalarMultCore(
	float Scalar, matrix* M1, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t NumResultElements = CudaGetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		float NewValue = Scalar * CudaGetMatrixElement(M1, ResultIndex);		
		CudaSetMatrixElement(
			Result,
			ResultIndex,
			NewValue
		);
	}
}

__global__
void CudaMatrixScalarMultThread(float Scalar, matrix* M1, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaMatrixScalarMultCore(Scalar, M1, Result, Start, Stride);
}

void CudaMatrixScalarMult(float Scalar, matrix* M1, matrix* Result)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(M1), BlockSize, Device
	);
	CudaMatrixScalarMultThread<<<NumBlocks, BlockSize>>>(Scalar, M1, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaMatrixSubtractCore(
	matrix* M1, matrix* M2, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t NumResultElements = CudaGetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		CudaSetMatrixElement(
			Result,
			ResultIndex,
			CudaGetMatrixElement(M1, ResultIndex) - 
			CudaGetMatrixElement(M2, ResultIndex)
		);
	}
}

__global__
void CudaMatrixSubtractThread(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaMatrixSubtractCore(M1, M2, Result, Start, Stride);
}

void CudaMatrixSubtract(matrix* M1, matrix* M2, matrix* Result)
{
	assert(M1->NumRows == M2->NumRows);
	assert(M1->NumColumns == M2->NumColumns);

	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(M1), BlockSize, BlockSize
	);

	CudaMatrixSubtractThread<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaMatrixScalarMultCoreColStride(
	float Scalar, matrix* M1, matrix* Result, uint32_t Start, uint32_t Stride
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

__device__
void CudaMatrixMeanCore(
	matrix* M1, matrix* Result, uint32_t Start, uint32_t Stride
)
{
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

__global__
void CudaMatrixMeanThread(matrix* M1, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaMatrixMeanCore(M1, Result, Start, Stride);
}

void CudaMatrixMean(matrix* M1, matrix* Result)
{
	/*NOTE:
	This function finds the sum of all the row vectors of matrix M1 and divides
	that sum by the number of rows. 

	M1 Dimensions: N x M
	Result Dimensions: 1 x M
	*/
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(M1->NumColumns, BlockSize, Device);
	CudaMatrixMeanThread<<<NumBlocks, BlockSize>>>(M1, Result);
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

void CudaFreeDenseLayer(dense_layer* DenseLayer)
{
	CudaFreeMatrixData(&DenseLayer->Weights);
	CudaFreeMatrixData(&DenseLayer->Bias);
	cudaFree(DenseLayer);
}

__device__
void CudaDenseForwardCore(
	matrix* Inputs,
	dense_layer* DenseLayer,
	matrix* Results,
	uint32_t Start,
	uint32_t Stride
)
{
	CudaMatrixMultCore(Inputs, &DenseLayer->Weights, Results, Start, Stride);
	CudaAddVectorToRowsCore(
		Results, &DenseLayer->Bias, Results, Start, Stride
	);
}

__global__
void CudaDenseForwardThread(
	matrix* Inputs, dense_layer* DenseLayer, matrix* Results
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaDenseForwardCore(Inputs, DenseLayer, Results, Start, Stride);
}

void CudaDenseForward(matrix* Inputs, dense_layer* DenseLayer, matrix* Results)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		DenseLayer->Weights.NumRows, BlockSize, Device
	);
	CudaDenseForwardThread<<<NumBlocks, BlockSize>>>(
		Inputs, DenseLayer, Results
	);
	cudaDeviceSynchronize();
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

void CudaFreeDenseLayerTrain(dense_layer_train_data* TrainData)
{
	CudaFreeMatrixData(&TrainData->WeightsDelta);
	CudaFreeMatrixData(&TrainData->BiasDelta);
	CudaFreeMatrixData(&TrainData->LayerGradient);
	cudaFree(TrainData);
}

__device__
void CudaDenseBackCore(
	matrix* Inputs,
	matrix* NextLayerGradient,
	dense_layer* DenseLayer,
	dense_layer_train_data* TrainData,
	uint32_t Start,
	uint32_t Stride
)
{
	CudaMatrixMultM2TransposeCore(
		NextLayerGradient,
		&DenseLayer->Weights,
		&TrainData->LayerGradient,
		Start,
		Stride
	);
	__syncthreads();

	CudaMatrixMultM1TransposeCore(
		Inputs, NextLayerGradient, &TrainData->WeightsDelta, Start, Stride
	);
	__syncthreads();
	
	CudaMatrixScalarMultCore(
		TrainData->LearningRate,
		&TrainData->WeightsDelta,
		&TrainData->WeightsDelta,
		Start,
		Stride
	);
	__syncthreads();
	
	CudaMatrixAddCore(
		&DenseLayer->Weights,
		&TrainData->WeightsDelta,
		&DenseLayer->Weights,
		Start,
		Stride
	);
	__syncthreads();
	
	CudaMatrixMeanCore(NextLayerGradient, &TrainData->BiasDelta, Start, Stride);
	__syncthreads();

	CudaMatrixScalarMultCore(
		TrainData->LearningRate,
		&TrainData->BiasDelta,
		&TrainData->BiasDelta,
		Start,
		Stride
	);
	__syncthreads();

	CudaMatrixAddCore(
		&DenseLayer->Bias,
		&TrainData->BiasDelta,
		&DenseLayer->Bias,
		Start,
		Stride
	);
	__syncthreads();
}

__global__
void CudaDenseBackThread(
	matrix* Inputs,
	matrix* NextLayerGradient,
	dense_layer* DenseLayer,
	dense_layer_train_data* TrainData
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = gridDim.x * blockDim.x;

	CudaDenseBackCore(
		Inputs, NextLayerGradient, DenseLayer, TrainData, Start, Stride
	);
}

void CudaDenseBack(
	matrix* Inputs,
	matrix* NextLayerGradient,
	dense_layer* DenseLayer,
	dense_layer_train_data* TrainData
)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(Inputs->NumRows, BlockSize, Device);
	CudaDenseBackThread<<<NumBlocks, BlockSize>>>(
		Inputs,
		NextLayerGradient,
		DenseLayer,
		TrainData
	);
	cudaDeviceSynchronize();
}

void CudaAllocReluTrain(
	relu_train_data** Result, uint32_t BatchSize, uint32_t InputDim
)
{
	cudaMallocManaged(Result, sizeof(relu_train_data));
	relu_train_data* TrainData = *Result;
	*TrainData = {};
	CudaInitMatrix(&TrainData->LayerGradient, BatchSize, InputDim);
}

// TODO: implement free
// void FreeReluTrain(relu_train_data* TrainData)
// {
// 	FreeMatrixData(TrainData->LayerGradient);
// 	free(TrainData);
// }

__device__
void CudaReluForwardCore(
	matrix* M1, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	for(uint32_t Row = Start; Row < M1->NumRows; Row += Stride)
	{
		for(uint32_t Col = 0; Col < M1->NumColumns; Col++)
		{
			float NewValue;
			float OldValue = CudaGetMatrixElement(M1, Row, Col);
			if(OldValue < 0)
			{
				NewValue = 0;
			}
			else
			{
				NewValue = OldValue;
			}
			CudaSetMatrixElement(Result, Row, Col, NewValue);
		}
	}
}

__global__
void CudaReluForwardThread(matrix* Inputs, matrix* Outputs)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaReluForwardCore(Inputs, Outputs, Start, Stride);
}

void CudaReluForward(matrix* Inputs, matrix* Outputs)
{
	assert(Inputs->NumRows == Outputs->NumRows);
	assert(Inputs->NumColumns == Outputs->NumColumns);

	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(Inputs->NumRows, BlockSize, Device);
	CudaReluForwardThread<<<NumBlocks, BlockSize>>>(Inputs, Outputs);
	cudaDeviceSynchronize();
}

__device__
void CudaReluBackCore(
	matrix* Inputs,
	matrix* NextLayerGradient,
	matrix* LayerGradient,
	uint32_t Start,
	uint32_t Stride
)
{
	for(uint32_t Row = Start; Row < Inputs->NumRows; Row += Stride)
	{
		for(uint32_t Col = 0; Col < Inputs->NumColumns; Col++)
		{
			float LayerGradientElement;
			float InputValue = CudaGetMatrixElement(Inputs, Row, Col);
			if(InputValue <= 0)
			{
				LayerGradientElement = 0;
			}
			else
			{
				LayerGradientElement = CudaGetMatrixElement(
					NextLayerGradient, Row, Col
				);
			}
			CudaSetMatrixElement(LayerGradient, Row, Col, LayerGradientElement);
		}
	}
}

__global__
void CudaReluBackThread(
	matrix* Inputs, matrix* NextLayerGradient, matrix* LayerGradient
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = gridDim.x * blockDim.x;
	CudaReluBackCore(
		Inputs,
		NextLayerGradient,
		LayerGradient,
		Start,
		Stride
	);
}

void CudaReluBack(
	matrix* Inputs, matrix* NextLayerGradient, relu_train_data* TrainData
)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(Inputs->NumRows, BlockSize, Device);
	CudaReluBackThread<<<NumBlocks, BlockSize>>>(
		Inputs, NextLayerGradient, &TrainData->LayerGradient
	);
	cudaDeviceSynchronize();
}

struct mse_layer
{
	uint32_t MaxThreads;
	float* SquaredErrorResults;
};

void CudaAllocMeanSquared(mse_layer** Result, uint32_t MaxThreads)
{
	cudaMallocManaged(Result, sizeof(mse_layer));
	mse_layer* Layer = *Result;
	*Layer = {};
	Layer->MaxThreads = MaxThreads;
	cudaMallocManaged(&Layer->SquaredErrorResults, MaxThreads * sizeof(float));
}

__device__
void CudaMeanSquaredForwardCore(
	float* SquaredErrorResults,
	matrix* Predictions,
	matrix* Labels,
	uint32_t Start,
	uint32_t Stride
)
{
	float Result = 0.0f;
	for(uint32_t Row = Start; Row < Predictions->NumRows; Row += Stride)
	{
		for(uint32_t Col = 0; Col < Predictions->NumColumns; Col++)
		{
			float Difference = (
				CudaGetMatrixElement(Predictions, Row, Col) - 
				CudaGetMatrixElement(Labels, Row, Col)
			);
			Result += Difference * Difference;
		}
	}
	float* SquaredError = SquaredErrorResults + Start;
	*SquaredError = Result;
}

__global__
void CudaMeanSquaredForwardThread(
	float* SquaredErrorResults, matrix* Predictions, matrix* Labels
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = gridDim.x * blockDim.x;

	CudaMeanSquaredForwardCore(
		SquaredErrorResults, Predictions, Labels, Start, Stride
	);		
}

__device__ 
float CudaCalculateMse(mse_layer* Layer, matrix* Predictions)
{
	int NumThreadsRan; 
	if(Layer->MaxThreads < Predictions->NumRows)
	{
		NumThreadsRan = Layer->MaxThreads;
	}
	else
	{
		NumThreadsRan = Predictions->NumRows;
	}

	float Sum = 0;
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < NumThreadsRan;
		ThreadIndex++
	)
	{
		float* SquaredError = Layer->SquaredErrorResults + ThreadIndex;
		Sum += *SquaredError;
	}

	// NOTE: this definition of MSE with a two in the denominator helps cancel 
	// CONT: out a two in the back derivation 
	float Mean = Sum / (2 * Predictions->NumRows);
	return Mean;
}

float CalculateMse(mse_layer* Layer, matrix* Predictions)
{
	int NumThreadsRan; 
	if(Layer->MaxThreads < Predictions->NumRows)
	{
		NumThreadsRan = Layer->MaxThreads;
	}
	else
	{
		NumThreadsRan = Predictions->NumRows;
	}

	float Sum = 0;
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < NumThreadsRan;
		ThreadIndex++
	)
	{
		float* SquaredError = Layer->SquaredErrorResults + ThreadIndex;
		Sum += *SquaredError;
	}

	// NOTE: this definition of MSE with a two in the denominator helps cancel 
	// CONT: out a two in the back derivation 
	float Mean = Sum / (2 * Predictions->NumRows);
	return Mean;
}

float CudaMeanSquaredForward(
	mse_layer* Layer, matrix* Predictions, matrix* Labels
)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(Predictions->NumRows, BlockSize, Device);
	assert((NumBlocks * BlockSize) < Layer->MaxThreads);
	memset(Layer->SquaredErrorResults, 0, Layer->MaxThreads * sizeof(float));
	CudaMeanSquaredForwardThread<<<NumBlocks, BlockSize>>>(
		Layer->SquaredErrorResults, Predictions, Labels
	);
	cudaDeviceSynchronize();

	return CalculateMse(Layer, Predictions);
}

void CudaAllocMseTrainData(
	mse_train_data** Result, uint32_t BatchSize, uint32_t PredictionDim
)
{
	cudaMallocManaged(Result, sizeof(mse_train_data));
	mse_train_data* TrainData = *Result;
	*TrainData = {};
	CudaInitMatrix(&TrainData->LayerGradient, BatchSize, PredictionDim);
}

// TODO: implement me
// void FreeMseTrainData(mse_train_data* TrainData)
// {
// 	FreeMatrixData(TrainData->LayerGradient);
// 	free(TrainData);
// }

__device__
void CudaMseBackCore(
	matrix* Predictions,
	matrix* Labels,
	mse_train_data* TrainData,
	uint32_t Start,
	uint32_t Stride
)
{
	CudaMatrixSubtractCore(
		Labels, Predictions, &TrainData->LayerGradient, Start, Stride
	);
	CudaMatrixScalarMultCore(
		1.0f / Predictions->NumColumns,
		&TrainData->LayerGradient,
		&TrainData->LayerGradient,
		Start,
		Stride
	);
}

__global__
void CudaMseBackThread(
	matrix* Predictions, matrix* Labels, mse_train_data* TrainData
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaMseBackCore(Predictions, Labels, TrainData, Start, Stride);
}

void CudaMeanSquaredBack(
	matrix* Predictions, matrix* Labels, mse_train_data* TrainData
)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(Predictions->NumRows, BlockSize, Device);
	CudaMseBackThread<<<NumBlocks, BlockSize>>>(Predictions, Labels, TrainData);
	cudaDeviceSynchronize();
}

void CudaAllocNeuralNet(
	neural_net** Result, uint32_t BatchSize, uint32_t InputDim
)
{
	cudaMallocManaged(Result, sizeof(neural_net));
	neural_net* NeuralNet = *Result;
	*NeuralNet = {};
	NeuralNet->BatchSize = BatchSize;
	NeuralNet->InputDim = InputDim;
}

uint32_t CudaAddLayerLink(neural_net* NeuralNet, layer_type LayerType)
{
	layer_link* LayerLink = NULL;
	cudaMallocManaged(&LayerLink, sizeof(layer_link));

	*LayerLink = {};
	LayerLink->Type = LayerType;
	uint32_t InputDim = NeuralNet->LastLink->Output->NumColumns;
	NeuralNet->LastLink->Next = LayerLink;
	LayerLink->Previous = NeuralNet->LastLink;
	LayerLink->Next = NULL;
	NeuralNet->LastLink = LayerLink;
	NeuralNet->NumLayers++;

	return InputDim;
}

// void FreeLayerLink(layer_link* LayerLink)
// {
// 	if(LayerLink->Output != NULL)
// 	{
// 		FreeMatrix(LayerLink->Output);
// 	}
// 	free(LayerLink);
// }

void CudaAddDense(
	neural_net* NeuralNet, uint32_t OutputDim, dense_layer* DenseLayer = NULL
)
{
	layer_link* LayerLink = NULL;
	cudaMallocManaged(&LayerLink, sizeof(layer_link));

	*LayerLink = {};
	uint32_t InputDim;
	if(NeuralNet->NumLayers == 0)
	{
		InputDim = NeuralNet->InputDim;
		NeuralNet->FirstLink = LayerLink;
		NeuralNet->LastLink = LayerLink;
		LayerLink->Next = NULL;
		LayerLink->Previous = NULL;
	}
	else
	{
		InputDim = NeuralNet->LastLink->Output->NumColumns;
		LayerLink->Previous = NeuralNet->LastLink;
		NeuralNet->LastLink->Next = LayerLink;
		NeuralNet->LastLink = LayerLink;
	}

	LayerLink->Type = LayerType_Dense;
	if(DenseLayer)
	{
		LayerLink->Data = DenseLayer;
	}
	else
	{
		CudaAllocDenseLayer(
			(dense_layer**) &LayerLink->Data, 
			InputDim,
			OutputDim
		);
	}
	CudaAllocMatrix(&LayerLink->Output, NeuralNet->BatchSize, OutputDim);

	NeuralNet->NumLayers++;
}

void CudaAddRelu(neural_net* NeuralNet)
{
	uint32_t InputDim = CudaAddLayerLink(NeuralNet, LayerType_Relu);
	layer_link* LayerLink = NeuralNet->LastLink;

	CudaAllocMatrix(&LayerLink->Output, NeuralNet->BatchSize, InputDim);
}

void CudaAddMeanSquared(neural_net* NeuralNet, uint32_t MaxThreads)
{
	CudaAddLayerLink(NeuralNet, LayerType_Mse);
	layer_link* LayerLink = NeuralNet->LastLink;

	CudaAllocMeanSquared((mse_layer**) &LayerLink->Data, MaxThreads);
}

void CudaResizedNeuralNet(
	neural_net** Result, neural_net* Source, uint32_t NewBatchSize
)
{
	// NOTE: this is needed b/c the result from each layer is preallocated
	// CONT: so we can't use different batch sizes with the same neural net.
	// CONT: Instead of copying all the data, I am using this function to 
	// CONT: create the new output matrices and reusing the dense_layer structs
	// CONT: from the Source net. This is a valuable approach for situations 
	// CONT: where you are testing in a loop, e.g. if you check the full-batch
	// CONT: loss after doing all the mini batches in an epoch. It's also a 
	// CONT: slightly smaller memory profile

	CudaAllocNeuralNet(Result, NewBatchSize, Source->InputDim);
	neural_net* NeuralNet = *Result;

	layer_link* LayerLink = Source->FirstLink;
	for(
		uint32_t LayerIndex = 0;
		LayerIndex < Source->NumLayers;
		LayerIndex++
	)
	{
		switch(LayerLink->Type)
		{
			case(LayerType_Dense):
			{
				dense_layer* DenseLayer = (dense_layer*) LayerLink->Data;
				CudaAddDense(
					NeuralNet, DenseLayer->Weights.NumColumns, DenseLayer
				);
				break;
			}
			case(LayerType_Relu):
			{
				CudaAddRelu(NeuralNet);
				break;
			}
			case(LayerType_Softmax):
			{
				// TODO: NOT IMPLEMENTED
				assert(false);
				break;
			}
			case(LayerType_CrossEntropy):
			{
				// TODO: NOT IMPLEMENTED
				assert(false);
				break;
			}
			case(LayerType_Mse):
			{
				CudaAddMeanSquared(NeuralNet, 1 << 14);
				break;
			}
			default:
			{				
				break;
			}
		}
		LayerLink = LayerLink->Next;
	}
}

__device__
void CudaNeuralNetForwardCore(
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	matrix** Predictions,
	uint32_t Start,
	uint32_t Stride
)
{
	matrix* Outputs = NULL;
	layer_link* LayerLink = NeuralNet->FirstLink;
	for(
		uint32_t LayerIndex = 0;
		LayerIndex < NeuralNet->NumLayers;
		LayerIndex++
	)
	{
		Outputs = LayerLink->Output;
		switch(LayerLink->Type)
		{
			case(LayerType_Dense):
			{
				CudaDenseForwardCore(
					Inputs,
					(dense_layer*) LayerLink->Data,
					Outputs,
					Start,
					Stride
				);
				break;
			}
			case(LayerType_Relu):
			{
				CudaReluForwardCore(Inputs, Outputs, Start, Stride);
				break;
			}
			case(LayerType_Softmax):
			{
				// TODO: not implemented
				break;
			}

			// NOTE: for NNs with loss layers, predictions must be captured 
			// CONT: with inputs the end of the loop since outputs 
			// CONT: will be updated to NULL
			case(LayerType_CrossEntropy):
			{
				// TODO: not implemented
				break;
			}
			case(LayerType_Mse):
			{
				if(Predictions)
				{
					*Predictions = Inputs;
				}
				if(Labels != NULL)
				{
					mse_layer* MseLayer = (mse_layer*) LayerLink->Data;
					CudaMeanSquaredForwardCore(
						MseLayer->SquaredErrorResults,
						Inputs,
						Labels,
						Start,
						Stride
					);
				}
				break;
			}

			default:
			{				
				break;
			}
		}
		Inputs = Outputs;
		LayerLink = LayerLink->Next;
	}

	if(Predictions != NULL && *Predictions == NULL)
	{
		// NOTE: if we didn't have a loss function, this is where we get the
		// CONT: predictions from
		*Predictions = Outputs;
	}
}

__global__
void CudaNeuralNetForwardThread(
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	matrix** Predictions
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;

	CudaNeuralNetForwardCore(
		NeuralNet,
		Inputs,
		Labels,
		Predictions,
		Start,
		Stride
	);
}

void CudaNeuralNetForward(
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	matrix** Predictions,
	float* LossResult
)
{
	// NOTE: *Predictions must be in cuda shared memory, since it is modified 
	// CONT: by the device side
	if(Predictions)
	{
		*Predictions = NULL;
	}

	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(Inputs->NumRows, BlockSize, Device);

	CudaNeuralNetForwardThread<<<NumBlocks, BlockSize>>>(
		NeuralNet,
		Inputs,
		Labels,
		Predictions
	);
	cudaDeviceSynchronize();

	if(LossResult)
	{
		layer_link* LayerLink = NeuralNet->LastLink;
		switch(LayerLink->Type)
		{
			case(LayerType_Mse):
			{
				mse_layer* MseLayer = (mse_layer*) LayerLink->Data;
				*LossResult = CalculateMse(MseLayer, *Predictions);
				break;
			}
			default:
			{
				assert(false);
				break;
			}
		}
	}
}

void CudaAllocNeuralNetTrainer(
	neural_net_trainer** Result,
	neural_net* NeuralNet,
	float LearningRate,
	layer_type LossLayer
)
{
	switch(LossLayer)
	{
		case(LayerType_Mse):
		{
			// TODO: figure out a better way to set up the max number of threads
			CudaAddMeanSquared(NeuralNet, 1 << 14);
			break;
		}
		case(LayerType_CrossEntropy):
		{
			// TODO: implement
			assert(false);
			break;
		}
		default:
		{
			break;
		}
	}

	cudaMallocManaged(Result, sizeof(neural_net_trainer));
	neural_net_trainer* Trainer = *Result;
	*Trainer = {};
	Trainer->NeuralNet = NeuralNet;
	cudaMallocManaged(
		&Trainer->TrainDataArray, NeuralNet->NumLayers * sizeof(void*)
	);
	void** TrainDataArray = Trainer->TrainDataArray;
	memset(TrainDataArray, 0, NeuralNet->NumLayers * sizeof(void*));
	
	layer_link* LayerLink = NeuralNet->FirstLink;
	for(
		uint32_t LayerIndex = 0;
		LayerIndex < NeuralNet->NumLayers;
		LayerIndex++
	)
	{
		switch(LayerLink->Type)
		{
			case(LayerType_Dense):
			{
				CudaAllocDenseLayerTrain(
					(dense_layer_train_data**) &TrainDataArray[LayerIndex],
					(dense_layer*) LayerLink->Data,
					LearningRate,
					NeuralNet->BatchSize
				);
				break;
			}
			case(LayerType_Relu):
			{
				layer_link* PreviousLayer = LayerLink->Previous;
				matrix* PrevOutputs = PreviousLayer->Output;
				CudaAllocReluTrain(
					(relu_train_data**) &TrainDataArray[LayerIndex],
					NeuralNet->BatchSize,
					PrevOutputs->NumColumns
				);
				break;
			}
			case(LayerType_Softmax):
			{
				// TODO: implement
				assert(false);
				break;
			}
			case(LayerType_CrossEntropy):
			{
				// TODO: implement
				assert(false);
				// layer_link* PreviousLayer = LayerLink->Previous;
				// softmax_layer* SoftmaxLayer = (softmax_layer*)(
				// 	PreviousLayer->Data
				// );

				// AllocCrossEntropySoftmaxTrain(
				// 	(
				// 		(cross_entropy_softmax_train_data**) 
				// 		&TrainDataArray[LayerIndex]
				// 	),
				// 	SoftmaxLayer
				// );
				break;
			}
			case(LayerType_Mse):
			{
				layer_link* PreviousLayer = LayerLink->Previous;
				matrix* PrevOutputs = PreviousLayer->Output;
				CudaAllocMseTrainData(
					(mse_train_data**) &TrainDataArray[LayerIndex],
					NeuralNet->BatchSize,
					PrevOutputs->NumColumns
				);
				break;
			}
			default:
			{				
				break;
			}
		}
		LayerLink = LayerLink->Next;
	}

	*Result = Trainer;
}

void CudaAllocNeuralNetTrainer(
	neural_net_trainer** Result,
	neural_net* NeuralNet,
	float LearningRate,
	layer_type LossLayer,
	uint32_t MiniBatchSize,
	uint32_t OutputDim
)
{
	// NOTE: function also allocates minibatch matrices
	CudaAllocNeuralNetTrainer(Result, NeuralNet, LearningRate, LossLayer);
	neural_net_trainer* Trainer = *Result;
	CudaAllocMatrix(&Trainer->MiniBatchData, MiniBatchSize, NeuralNet->InputDim);
	CudaAllocMatrix(&Trainer->MiniBatchLabels, MiniBatchSize, OutputDim);
}

__device__
void CudaTrainNeuralNetCore(
	neural_net_trainer* Trainer,
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	uint32_t Epochs,
	uint32_t Start,
	uint32_t Stride,
	bool PrintStatus = false
)
{
	layer_link* LayerLink;
	float Loss = -1.0f;
	for(uint32_t Epoch = 0; Epoch < Epochs; Epoch++)
	{
		matrix* Predictions = NULL;
		CudaNeuralNetForwardCore(
			NeuralNet,
			Inputs,
			Labels,
			&Predictions,
			Start,
			Stride
		);
		__syncthreads();

		if(Start == 0)
		{
			// NOTE: loss summation is single-threaded
			layer_link* LayerLink = NeuralNet->LastLink;
			switch(LayerLink->Type)
			{
				case(LayerType_Mse):
				{
					mse_layer* MseLayer = (mse_layer*) LayerLink->Data;
					Loss = CudaCalculateMse(MseLayer, Predictions);
					break;
				}
				// NOTE: can add more loss types here
				default:
				{
					// TODO: error logging
					break;
				}
			}
			if(PrintStatus)
			{
				printf("Epoch %d Loss: %f\n", Epoch, Loss);
			}
		}
		__syncthreads();

		matrix* NextLayerGradient = NULL;
		LayerLink = NeuralNet->LastLink;
		for(
			int32_t LayerIndex = ((int32_t) NeuralNet->NumLayers) - 1;
			LayerIndex >= 0;
			LayerIndex--
		)
		{
			void* TrainData = Trainer->TrainDataArray[LayerIndex];
			layer_link* PreviousLayer = LayerLink->Previous;
			matrix* LayerInputs;
			if(PreviousLayer != NULL)
			{
				LayerInputs = PreviousLayer->Output;
			}
			else
			{
				LayerInputs = Inputs;
			}
			switch(LayerLink->Type)
			{
				case(LayerType_Dense):
				{
					dense_layer_train_data* DenseTrain = (
						(dense_layer_train_data*) TrainData
					);
					CudaDenseBackCore(
						LayerInputs,
						NextLayerGradient,
						(dense_layer*) LayerLink->Data,
						DenseTrain,
						Start,
						Stride
					);
					NextLayerGradient = &DenseTrain->LayerGradient;
					break;
				}
				case(LayerType_Relu):
				{
					relu_train_data* ReluTrain = (relu_train_data*) TrainData;
					matrix* LayerGradient = &ReluTrain->LayerGradient;
					CudaReluBackCore(
						LayerInputs,
						NextLayerGradient,
						LayerGradient,
						Start,
						Stride
					);
					NextLayerGradient = LayerGradient;
					break;
				}
				case(LayerType_Softmax):
				{
					// TODO: implement
					// assert(false);
					break;
				}
				case(LayerType_Mse):
				{
					mse_train_data* MseTrain = (mse_train_data*) TrainData;

					CudaMseBackCore(
						Predictions,
						Labels,
						MseTrain,
						Start,
						Stride
					);
					NextLayerGradient = &MseTrain->LayerGradient;
					break;
				}
				case(LayerType_CrossEntropy):
				{
					// TODO: implement
					// cross_entropy_softmax_train_data* XEntropyTrain = (
					// 	(cross_entropy_softmax_train_data*) TrainData
					// );

					// CrossEntropySoftmaxBack(
					// 	MatrixOpJobs,
					// 	Predictions, 
					// 	Labels,
					// 	XEntropyTrain
					// );
					// NextLayerGradient = &XEntropyTrain->LayerGradient;
					break;
				}
				default:
				{
					break;
				}
			}
			LayerLink = PreviousLayer;
		}

		__syncthreads();
	}
}

__global__
void CudaTrainNeuralNetThread(
	neural_net_trainer* Trainer,
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	uint32_t Epochs
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = blockDim.x * gridDim.x;
	CudaTrainNeuralNetCore(
		Trainer,
		NeuralNet,
		Inputs,
		Labels,
		Epochs,
		Start,
		Stride
	);
}

void CudaTrainNeuralNet(
	neural_net_trainer* Trainer,
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	uint32_t Epochs,
	bool ShouldInitDenseLayers = true,
	bool PrintStatus = false
)
{
	if(ShouldInitDenseLayers)
	{
		InitDenseLayers(NeuralNet);
	}

	// TODO: get maximum number of threads we'll need across all layers instead of just setting it by batch size
	int Device = 0;
	int BlockSize = GetBlockSize(Device);
	int NumBlocks = GetNumBlocks(Inputs->NumRows, BlockSize, Device);
	CudaTrainNeuralNetThread<<<NumBlocks, BlockSize>>>(
		Trainer,
		NeuralNet,
		Inputs,
		Labels,
		Epochs
	);
	cudaDeviceSynchronize();
}

float CudaTopOneAccuracy(neural_net* NeuralNet, matrix* Inputs, matrix* Labels)
{
	matrix* Predictions = NULL;
	CudaNeuralNetForward(
		NeuralNet,
		Inputs,
		Labels,
		&Predictions,
		NULL
	);
	
	uint32_t TotalCorrect = 0;
	for(
		uint32_t SampleIndex = 0;
		SampleIndex < Predictions->NumRows;
		SampleIndex++
	)
	{
		uint32_t PredictedLabel = ArgMax(
			GetMatrixRow(Predictions, SampleIndex), Predictions->NumColumns
		);
		uint32_t ActualLabel = ArgMax(
			GetMatrixRow(Labels, SampleIndex), Predictions->NumColumns
		);
		if(PredictedLabel == ActualLabel)
		{
			TotalCorrect++;
		}
	}

	return (float) TotalCorrect / (float) Predictions->NumRows;
}

void CudaTrainNeuralNetMiniBatch(
	neural_net_trainer* Trainer,
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	uint32_t Epochs,
	bool ShouldInitDenseLayers = true,
	bool PrintStatus = false,
	float TrainingAccuracyThreshold = 1.1f,
	float LossThreshold = -1.0f,
	neural_net* FullBatchNnViewer = NULL
)
{
	// NOTE: Train with minibatches sampled from Inputs
	assert(Trainer->MiniBatchData != NULL);
	assert(Trainer->MiniBatchLabels != NULL);

	matrix* MiniBatchData = Trainer->MiniBatchData;
	matrix* MiniBatchLabels = Trainer->MiniBatchLabels;
	uint32_t TrainingSamples = Inputs->NumRows;
	uint32_t MiniBatchSize = MiniBatchData->NumRows;
	
	if(ShouldInitDenseLayers)
	{
		InitDenseLayers(NeuralNet);
	}

	int_shuffler IntShuffler = MakeIntShuffler(TrainingSamples);

	for(uint32_t Epoch = 0; Epoch < Epochs; Epoch++)
	{
		ShuffleInts(&IntShuffler);
		for(
			uint32_t BatchIndex = 0;
			BatchIndex < TrainingSamples / MiniBatchSize;
			BatchIndex++
		)
		{
			// NOTE: create mini batch
			uint32_t IndexHandleStart = BatchIndex * MiniBatchSize;
			for(
				uint32_t IndexHandle = IndexHandleStart;
				IndexHandle < (IndexHandleStart + MiniBatchSize);
				IndexHandle++
			)
			{
				int RowToGet = IntShuffler.Result[IndexHandle];
				float* DataRow = GetMatrixRow(Inputs, RowToGet);
				float* LabelsRow = GetMatrixRow(Labels, RowToGet);

				float* MiniBatchDataRow = GetMatrixRow(
					MiniBatchData, IndexHandle - IndexHandleStart
				);
				float* MiniBatchLabelRow = GetMatrixRow(
					MiniBatchLabels, IndexHandle - IndexHandleStart
				);

				memcpy(
					MiniBatchDataRow,
					DataRow,
					MiniBatchData->NumColumns * sizeof(float)
				);
				memcpy(
					MiniBatchLabelRow,
					LabelsRow,
					MiniBatchLabels->NumColumns * sizeof(float)
				);
			}

			// NOTE: train on mini batch
			CudaTrainNeuralNet(
				Trainer,
				NeuralNet,
				MiniBatchData,
				MiniBatchLabels,
				1,
				false,
				false
			);
		}

		float Loss = -1.0f;
		CudaNeuralNetForward(
			FullBatchNnViewer,
			Inputs,
			Labels,
			NULL,
			&Loss
		);
		float TrainingAccuracy = CudaTopOneAccuracy(
			FullBatchNnViewer, Inputs, Labels
		);
		if(PrintStatus)
		{
			printf(
				"Epoch %d Loss, Accuracy: %f, %f\n",
				Epoch,
				Loss,
				TrainingAccuracy
			);
		}
		if(
			TrainingAccuracy >= TrainingAccuracyThreshold ||
			Loss <= LossThreshold
		)
		{
			break;
		}
	}
}