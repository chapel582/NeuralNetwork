// TODO: handle cudaMallocManaged failures
// TODO: query max block size
#include "arg_max.h"
#include "int_shuffler.h"
#include "neural_net.h"
#include "matrix.h"

#include "neural_net.cpp"
#include "matrix.cpp"
#include "mnist_test.cpp"
#include "matrix_test.cpp"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

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
	// TODO: query for max block size?
	return (Range + BlockSize - 1) / BlockSize;
}

__device__
int CudaGetNumBlocks(int Range, int BlockSize)
{
	return (Range + BlockSize - 1) / BlockSize;
}

__device__
void CudaMatrixMultCore(
	matrix* M1, matrix* M2, matrix* Result, int Start, int Stride
)
{
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

__global__
void CudaMatrixMultThread(matrix* M1, matrix* M2, matrix* Result)
{	
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

	CudaMatrixMultCore(M1, M2, Result, Start, Stride);
}

void CudaMatrixMult(matrix* M1, matrix* M2, matrix* Result)
{
	assert(M1->NumColumns == M2->NumRows);
	// NOTE: not sure if this should be a variable or queried or tracked with 
	// CONT: a data structure
	int BlockSize = 256;

	// NOTE: NumBlocks is always at least one, and grows as the data to 
	// NOTE: process grows
	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize);
	CudaMatrixMultThread<<<NumBlocks, BlockSize>>>(M1, M2, Result);
	cudaDeviceSynchronize();
}

__device__
void CudaAddVectorToRowsCore(
	matrix* M1, matrix* Vector, matrix* Result, int Start, int Stride
)
{
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
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;

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
	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(M1->NumRows, BlockSize);
	CudaAddVectorToRowsThread<<<NumBlocks, BlockSize>>>(M1, Vector, Result);
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

__device__
void CudaDenseForwardCore(
	matrix* Inputs,
	dense_layer* DenseLayer,
	matrix* Results,
	int Start,
	int Stride
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
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	int Stride = blockDim.x * gridDim.x;

	CudaDenseForwardCore(Inputs, DenseLayer, Results, Start, Stride);
}

void CudaDenseForward(matrix* Inputs, dense_layer* DenseLayer, matrix* Results)
{
	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(DenseLayer->Weights.NumRows, BlockSize);
	CudaDenseForwardThread<<<NumBlocks, BlockSize>>>(
		Inputs, DenseLayer, Results
	);
	cudaDeviceSynchronize();
	// CudaMatrixMult(Inputs, &DenseLayer->Weights, Results);
	// CudaAddVectorToRows(Results, &DenseLayer->Bias, Results);	
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
void CudaReluForwardCore(matrix* M1, matrix* Result, int Start, int Stride)
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
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	int Stride = blockDim.x * gridDim.x;

	CudaReluForwardCore(Inputs, Outputs, Start, Stride);
}

void CudaReluForward(matrix* Inputs, matrix* Outputs)
{
	assert(Inputs->NumRows == Outputs->NumRows);
	assert(Inputs->NumColumns == Outputs->NumColumns);

	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(Inputs->NumRows, BlockSize);
	CudaReluForwardThread<<<NumBlocks, BlockSize>>>(Inputs, Outputs);
	cudaDeviceSynchronize();
}

__global__
void CudaReluBackCore(
	matrix* Inputs, matrix* NextLayerGradient, matrix* LayerGradient
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = gridDim.x * blockDim.x;
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

void CudaReluBack(
	matrix* Inputs, matrix* NextLayerGradient, relu_train_data* TrainData
)
{
	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(Inputs->NumRows, BlockSize);
	CudaReluBackCore<<<NumBlocks, BlockSize>>>(
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
	int Start,
	int Stride
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
	int Start = blockIdx.x * blockDim.x + threadIdx.x;
	int Stride = gridDim.x * blockDim.x;

	CudaMeanSquaredForwardCore(
		SquaredErrorResults, Predictions, Labels, Start, Stride
	);		
}

float CudaMeanSquaredForward(
	mse_layer* Layer, matrix* Predictions, matrix* Labels
)
{
	int BlockSize = 256;
	int NumBlocks = GetNumBlocks(Predictions->NumRows, BlockSize);
	assert((NumBlocks * BlockSize) < Layer->MaxThreads);
	memset(Layer->SquaredErrorResults, 0, Layer->MaxThreads * sizeof(float));
	CudaMeanSquaredForwardThread<<<NumBlocks, BlockSize>>>(
		Layer->SquaredErrorResults, Predictions, Labels
	);
	cudaDeviceSynchronize();
		
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

void CudaMeanSquaredBack(
	matrix* Predictions, matrix* Labels, mse_train_data* TrainData
)
{
	CudaMatrixSubtract(
		Labels, Predictions, &TrainData->LayerGradient
	);
	CudaMatrixScalarMult(
		1.0f / Predictions->NumColumns,
		&TrainData->LayerGradient,
		&TrainData->LayerGradient
	);
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

void CudaNeuralNetForward(
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	matrix** Predictions,
	float* LossResult
)
{
	if(Predictions)
	{
		*Predictions = NULL;
	}
	matrix* Outputs = NULL;
	float Loss = -1.0f;
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
				CudaDenseForward(
					Inputs,
					(dense_layer*) LayerLink->Data,
					Outputs
				);
				break;
			}
			case(LayerType_Relu):
			{
				CudaReluForward(Inputs, Outputs);
				break;
			}
			case(LayerType_Softmax):
			{
				assert(false);
				break;
			}

			// NOTE: for NNs with loss layers, predictions must be captured 
			// CONT: with inputs the end of the loop since outputs 
			// CONT: will be updated to NULL
			case(LayerType_CrossEntropy):
			{
				assert(false);
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
					Loss = CudaMeanSquaredForward(
						(mse_layer*) LayerLink->Data, Inputs, Labels
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

	if(LossResult)
	{
		*LossResult = Loss;
	}
	if(Predictions != NULL && *Predictions == NULL)
	{
		// NOTE: if we didn't have a loss function, this is where we get the
		// CONT: predictions from
		*Predictions = Outputs;
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

	layer_link* LayerLink;
	float Loss = -1.0f;
	for(uint32_t Epoch = 0; Epoch < Epochs; Epoch++)
	{
		matrix* Predictions = NULL;
		CudaNeuralNetForward(
			NeuralNet,
			Inputs,
			Labels,
			&Predictions,
			&Loss
		);
		if(PrintStatus)
		{
			printf("Epoch %d Loss: %f\n", Epoch, Loss);
		}

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
					CudaDenseBack(
						LayerInputs,
						NextLayerGradient,
						(dense_layer*) LayerLink->Data,
						DenseTrain
					);
					NextLayerGradient = &DenseTrain->LayerGradient;
					break;
				}
				case(LayerType_Relu):
				{
					relu_train_data* ReluTrain = (relu_train_data*) TrainData;
					CudaReluBack(
						LayerInputs,
						NextLayerGradient,
						ReluTrain
					);
					NextLayerGradient = &ReluTrain->LayerGradient;
					break;
				}
				case(LayerType_Softmax):
				{
					assert(false);
					break;
				}
				case(LayerType_Mse):
				{
					mse_train_data* MseTrain = (mse_train_data*) TrainData;

					CudaMeanSquaredBack(
						Predictions,
						Labels,
						MseTrain
					);
					NextLayerGradient = &MseTrain->LayerGradient;
					break;
				}
				case(LayerType_CrossEntropy):
				{
					// TODO: implement
					assert(false);
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
	}
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

#define SAVE_RESULTS 0
matrix* TestMatrixResult(
	matrix* M1,
	char* FilePathBuffer,
	size_t FilePathBufferSize,
	char* TestDataDirectory,
	const char* TestName,
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

void TestFloatResult(
	float Result,
	char* FilePathBuffer,
	size_t FilePathBufferSize,
	char* TestDataDirectory,
	const char* TestName,
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
		TestMatrixResult(
			MultResult,
			FilePathBuffer,
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMultResult",
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
		TestMatrixResult(
			MultResult2,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaNonSquareMult",
			EndianString
		);

		matrix* AddResult;
		CudaAllocMatrix(&AddResult, M1->NumRows, M1->NumColumns);
		CudaMatrixAdd(M1, M2, AddResult);
		TestMatrixResult(
			AddResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMatrixAdd",
			EndianString
		);

		matrix* AddVectorResult;
		CudaAllocMatrix(&AddVectorResult, M1->NumRows, M1->NumColumns);
		matrix* Vector;
		CudaAllocMatrix(&Vector, 1, M1->NumColumns);
		FillMatrixConsecutive(Vector);
		CudaAddVectorToRows(M1, Vector, AddVectorResult);
		TestMatrixResult(
			AddVectorResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaAddVectorToRows",
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
		TestMatrixResult(
			M5TMultResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMatrixMultM1Transpose",
			EndianString
		);

		MatrixClear(M5TMultResult);
		SetMatrixElement(M6, 0, 1, 7);
		SetMatrixElement(M6, 1, 2, 13);
		
		CudaMatrixMultM1Transpose(M5, M6, M5TMultResult);
		TestMatrixResult(
			M5TMultResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaNonSymmetricMatrixMultM1Transpose",
			EndianString
		);

		matrix* M6TMultResult;
		CudaAllocM2TransposeMultResultMatrix(&M6TMultResult, M5, M6);

		CudaMatrixMultM2Transpose(M5, M6, M6TMultResult);
		TestMatrixResult(
			M6TMultResult,
			FilePathBuffer,
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMatrixMultM2Transpose",
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
		TestMatrixResult(
			M7TM8TMultResult,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMatrixMultM1M2Transpose",
			EndianString
		);

		matrix* M9;
		NumRows = 3;
		NumColumns = 4;
		CudaAllocMatrix(&M9, NumRows, NumColumns);
		FillMatrixConsecutive(M9);
		
		CudaMatrixScalarMult(0.5f, M9, M9);
		TestMatrixResult(
			M9,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMatrixScalarMult",
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
		TestMatrixResult(
			M10Mean,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMatrixRowMean",
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

		TestMatrixResult(
			SubResult,
			FilePathBuffer,
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMatrixSub",
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
		TestMatrixResult(
			Outputs,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaForwardDense",
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
		TestMatrixResult(
			&DenseLayer->Weights,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaDenseWeightsAfterUpdate",
			EndianString
		);
		TestMatrixResult(
			&DenseLayer->Bias,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaDenseBiasAfterUpdate",
			EndianString
		);
		TestMatrixResult(
			&TrainData->LayerGradient,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaDenseLayerGradient",
			EndianString
		);
	}
	// SECTION STOP: Dense layer tests

	// SECTION START: RELU tests
	{
		uint32_t BatchSize = 8;
		uint32_t InputDim = 4;

		matrix* Inputs;
		CudaAllocMatrix(&Inputs, BatchSize, InputDim);
		FillMatrixConsecutive(Inputs);

		matrix* Outputs;
		CudaAllocMatrix(&Outputs, BatchSize, InputDim);
		CudaReluForward(Inputs, Outputs);
		TestMatrixResult(
			Outputs,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaReluForwardPositive",
			EndianString
		);

		matrix* NextLayerGradient;
		CudaAllocMatrix(&NextLayerGradient, BatchSize, InputDim);
		FillMatrixConsecutive(NextLayerGradient);

		relu_train_data* TrainData;
		CudaAllocReluTrain(&TrainData, BatchSize, InputDim);
		CudaReluBack(Inputs, NextLayerGradient, TrainData);
		TestMatrixResult(
			&TrainData->LayerGradient,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaReluLayerGradientPositive",
			EndianString
		);

		CudaMatrixScalarMult(-1.0f, Inputs, Inputs);
		CudaReluForward(Inputs, Outputs);
		TestMatrixResult(
			Outputs,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaReluForwardNegative",
			EndianString
		);

		CudaReluBack(Inputs, NextLayerGradient, TrainData);
		TestMatrixResult(
			&TrainData->LayerGradient,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaReluLayerGradientNegative",
			EndianString
		);
	}
	// SECTION STOP: RELU Tests

	// SECTION START: MSE Test
	{
		uint32_t BatchSize = 8;
		uint32_t NumClasses = 4;

		matrix* Predictions = NULL;
		CudaAllocMatrix(&Predictions, BatchSize, NumClasses);
		FillOneHotMatrix(Predictions);
		
		matrix* Labels = NULL; 
		CudaAllocMatrix(&Labels, BatchSize, NumClasses);
		FillOneHotMatrix(Labels);

		mse_layer* MseLayer = NULL;
		CudaAllocMeanSquared(&MseLayer, 1 << 14);

		float Loss = CudaMeanSquaredForward(MseLayer, Predictions, Labels);
		TestFloatResult(
			Loss,
			FilePathBuffer,
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMSELoss",
			EndianString
		);

		mse_train_data* TrainData = NULL;
		CudaAllocMseTrainData(&TrainData, BatchSize, NumClasses);
		CudaMeanSquaredBack(Predictions, Labels, TrainData);
		TestMatrixResult(
			&TrainData->LayerGradient,
			FilePathBuffer, 
			sizeof(FilePathBuffer),
			TestDataDirectory,
			"CudaMSEBackOK",
			EndianString
		);
	}
	// SECTION STOP: MSE Test

	// // TODO: maybe add another MSE test with non-zero resulting loss and 
	// // CONT: layer gradient

	// // SECTION START: Linear NN test
	// {
	// 	uint32_t BatchSize = 10;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(
	// 		&NeuralNet, BatchSize, InputDim
	// 	);
	// 	CudaAddDense(NeuralNet, 1);
	// 	CudaAddDense(NeuralNet, 1);
	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	SetMatrixElement(&DenseLayer->Weights, 0, 0, 2);
	// 	SetMatrixElement(&DenseLayer->Bias, 0, 0, 1);
	// 	DenseLayer = (dense_layer*) NeuralNet->LastLink->Data;
	// 	SetMatrixElement(&DenseLayer->Weights, 0, 0, 3);
	// 	SetMatrixElement(&DenseLayer->Bias, 0, 0, 1);
	// 	// NOTE: should be equivalent to 6x + 4

	// 	matrix* Predictions = NULL;
	// 	CudaNeuralNetForward(
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
	// 		"CudaLinearForwardNN",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Linear NN test

	// // SECTION START: Dim loss NN test
	// {
	// 	uint32_t BatchSize = 4;
	// 	uint32_t InputDim = 4;

	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, 2);
	// 	CudaAddDense(NeuralNet, 1);
	// 	dense_layer* DenseLayer1 = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	FillMatrixConsecutive(&DenseLayer1->Weights);
	// 	FillMatrixConsecutive(&DenseLayer1->Bias);
	// 	dense_layer* DenseLayer2 = (dense_layer*) NeuralNet->LastLink->Data;
	// 	FillMatrixConsecutive(&DenseLayer2->Weights);
	// 	FillMatrixConsecutive(&DenseLayer2->Bias);

	// 	matrix* Predictions = NULL;
	// 	CudaNeuralNetForward(
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
	// 		"CudaDimReductionLinearForwardNN",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Dim loss NN test

	// // SECTION START: Positive Relu NN test
	// {
	// 	uint32_t BatchSize = 10;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, 1);
	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	SetMatrixElement(&DenseLayer->Weights, 0, 0, 2);
	// 	SetMatrixElement(&DenseLayer->Bias, 0, 0, 1);
	// 	CudaAddRelu(NeuralNet);
	// 	// NOTE: should be equivalent to 2x + 1

	// 	matrix* Predictions = NULL;
	// 	CudaNeuralNetForward(
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
	// 		"CudaPosReluNN",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Positive Relu NN test

	// // SECTION START: Negative Relu NN test
	// {
	// 	uint32_t BatchSize = 10;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixNegativeConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, 1);
	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	SetMatrixElement(&DenseLayer->Weights, 0, 0, 2);
	// 	SetMatrixElement(&DenseLayer->Bias, 0, 0, 1);
	// 	CudaAddRelu(NeuralNet);
	// 	// NOTE: should be equivalent to 2x, but then everything is zeroed due
	// 	// CONT: to RELU

	// 	matrix* Predictions = NULL;
	// 	CudaNeuralNetForward(
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
	// 		"CudaNegReluNN",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: Negative Relu NN test

	// // SECTION START: One neuron training
	// {
	// 	uint32_t BatchSize = 5;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs = NULL;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, 1);

	// 	// NOTE: should be equivalent to 2x + 1
	// 	matrix* Labels;
	// 	CudaAllocMatrix(&Labels, BatchSize, 1);
	// 	SetMatrixElement(Labels, 0, 0, 3);
	// 	SetMatrixElement(Labels, 1, 0, 5);
	// 	SetMatrixElement(Labels, 2, 0, 7);
	// 	SetMatrixElement(Labels, 3, 0, 9);
	// 	SetMatrixElement(Labels, 4, 0, 11);

	// 	neural_net_trainer* Trainer;
	// 	CudaAllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	CudaTrainNeuralNet(
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
	// 		"CudaOneNeuronNN_Weights",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Bias,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"CudaOneNeuronNN_Bias",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: One neuron training

	// // SECTION START: More one neuron training
	// {
	// 	uint32_t BatchSize = 6;
	// 	uint32_t InputDim = 1;

	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, 1);

	// 	// NOTE: should be equivalent to 5x - 3
	// 	matrix* Labels;
	// 	CudaAllocMatrix(&Labels, BatchSize, 1);
	// 	SetMatrixElement(Labels, 0, 0, 2);
	// 	SetMatrixElement(Labels, 1, 0, 7);
	// 	SetMatrixElement(Labels, 2, 0, 12);
	// 	SetMatrixElement(Labels, 3, 0, 17);
	// 	SetMatrixElement(Labels, 4, 0, 22);
	// 	SetMatrixElement(Labels, 5, 0, 27);

	// 	neural_net_trainer* Trainer;
	// 	CudaAllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	CudaTrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		1000
	// 	);

	// 	dense_layer* DenseLayer = (dense_layer*) NeuralNet->FirstLink->Data;
	// 	TestMatrixResult(
	// 		&DenseLayer->Weights,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"CudaOneNeuronNN_Weights_2",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Bias,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"CudaOneNeuronNN_Bias_2",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: More one neuron training

	// // SECTION START: threaded two neuron training
	// {
	// 	uint32_t BatchSize = 2;
	// 	uint32_t InputDim = 2;
	// 	uint32_t OutputDim = 2;

	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, OutputDim);

	// 	// NOTE: Labels set up to converge weight to 
	// 	/* CONT: 
	// 		W = 
	// 			| 2 3 |
	// 			| 4 5 |
	// 		b = 
	// 			| 1 2 |
	// 	*/
	// 	matrix* Labels;
	// 	CudaAllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	SetMatrixElement(Labels, 0, 0, 11);
	// 	SetMatrixElement(Labels, 0, 1, 15);

	// 	SetMatrixElement(Labels, 1, 0, 23);
	// 	SetMatrixElement(Labels, 1, 1, 31);

	// 	neural_net_trainer* Trainer;
	// 	CudaAllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	CudaTrainNeuralNet(
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
	// 		"CudaTwoNeuronNN_Weights",
	// 		EndianString
	// 	);
	// 	TestMatrixResult(
	// 		&DenseLayer->Bias,
	// 		FilePathBuffer, 
	// 		sizeof(FilePathBuffer),
	// 		TestDataDirectory,
	// 		"CudaTwoNeuronNN_Bias",
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
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, OutputDim);

	// 	// NOTE: Labels set up to converge weight to 
	// 	/* CONT: 
	// 		W = 
	// 			| 2 3 |
	// 			| 4 5 |
	// 		b = 
	// 			| 1 2 |
	// 	*/
	// 	matrix* Labels;
	// 	CudaAllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	SetMatrixElement(Labels, 0, 0, 11);
	// 	SetMatrixElement(Labels, 0, 1, 15);

	// 	SetMatrixElement(Labels, 1, 0, 23);
	// 	SetMatrixElement(Labels, 1, 1, 31);

	// 	neural_net_trainer* Trainer;
	// 	CudaAllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	CudaTrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100
	// 	);

	// 	matrix* Predictions = NULL;
	// 	CudaNeuralNetForward(
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
	// 		"CudaLinearOneLayerPrediction",
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
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	FillMatrixConsecutive(Inputs);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, HiddenDim);
	// 	CudaAddDense(NeuralNet, OutputDim);

	// 	matrix* Labels;
	// 	CudaAllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	SetMatrixElement(Labels, 0, 0, 11);
	// 	SetMatrixElement(Labels, 0, 1, 15);

	// 	SetMatrixElement(Labels, 1, 0, 23);
	// 	SetMatrixElement(Labels, 1, 1, 31);

	// 	neural_net_trainer* Trainer;
	// 	CudaAllocNeuralNetTrainer(
	// 		&Trainer,
	// 		NeuralNet,
	// 		0.01f,
	// 		LayerType_Mse
	// 	);

	// 	CudaTrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100
	// 	);

	// 	matrix* Predictions = NULL;
	// 	CudaNeuralNetForward(
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
	// 		"CudaLinearTwoLayerPrediction",
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
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	SetMatrixElement(Inputs, 0, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 0, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 1, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 1, 1, 1.0f);

	// 	SetMatrixElement(Inputs, 2, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 2, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 3, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 3, 1, 1.0f);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, HiddenDim);
	// 	CudaAddRelu(NeuralNet);
	// 	CudaAddDense(NeuralNet, OutputDim);

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
	// 	CudaNeuralNetForward(
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
	// 		"CudaGoodXorForward",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: XOR Forward

	// 	// SECTION START: forward XOR with good initial weights
	// {
	// 	// NOTE: in keras, 8 neurons and one dense layer + RELU seems 
	// 	// CONT: to work. no momentum needed for high-dimensional stuff 
	// 	// CONT: 2000 epochs were needed too
	// 	uint32_t BatchSize = 4;
	// 	uint32_t InputDim = 2;
	// 	uint32_t HiddenDim = 8;
	// 	uint32_t OutputDim = 1;

	// 	matrix* Inputs;
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	SetMatrixElement(Inputs, 0, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 0, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 1, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 1, 1, 1.0f);

	// 	SetMatrixElement(Inputs, 2, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 2, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 3, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 3, 1, 1.0f);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, HiddenDim);
	// 	CudaAddRelu(NeuralNet);
	// 	CudaAddDense(NeuralNet, OutputDim);

	// 	matrix* Labels;
	// 	CudaAllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	// NOTE: this is set up to converge to a xor function
	// 	SetMatrixElement(Labels, 0, 0, 0.0f);
	// 	SetMatrixElement(Labels, 1, 0, 1.0f);
	// 	SetMatrixElement(Labels, 2, 0, 1.0f);
	// 	SetMatrixElement(Labels, 3, 0, 0.0f);

	// 	neural_net_trainer* Trainer;
	// 	CudaAllocNeuralNetTrainer(
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

	// 	CudaTrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100,
	// 		false
	// 	);

	// 	matrix* Predictions = NULL;
	// 	CudaNeuralNetForward(
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
	// 		"CudaForwardXor_StaticTraining",
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
	// 	CudaAllocMatrix(&Inputs, BatchSize, InputDim);
	// 	SetMatrixElement(Inputs, 0, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 0, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 1, 0, 0.0f);
	// 	SetMatrixElement(Inputs, 1, 1, 1.0f);

	// 	SetMatrixElement(Inputs, 2, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 2, 1, 0.0f);

	// 	SetMatrixElement(Inputs, 3, 0, 1.0f);
	// 	SetMatrixElement(Inputs, 3, 1, 1.0f);

	// 	neural_net* NeuralNet = NULL;
	// 	CudaAllocNeuralNet(&NeuralNet, BatchSize, InputDim);
	// 	CudaAddDense(NeuralNet, HiddenDim);
	// 	CudaAddRelu(NeuralNet);
	// 	CudaAddDense(NeuralNet, OutputDim);

	// 	matrix* Labels;
	// 	CudaAllocMatrix(&Labels, BatchSize, OutputDim);
		
	// 	// NOTE: this is set up to converge to a xor function
	// 	SetMatrixElement(Labels, 0, 0, 0.0f);
	// 	SetMatrixElement(Labels, 1, 0, 1.0f);
	// 	SetMatrixElement(Labels, 2, 0, 1.0f);
	// 	SetMatrixElement(Labels, 3, 0, 0.0f);

	// 	neural_net_trainer* Trainer;
	// 	CudaAllocNeuralNetTrainer(
	// 		&Trainer, NeuralNet, 0.1f, LayerType_Mse
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

	// 	CudaTrainNeuralNet(
	// 		Trainer,
	// 		NeuralNet,
	// 		Inputs,
	// 		Labels,
	// 		100,
	// 		false
	// 	);

	// 	matrix* Predictions = NULL;
	// 	CudaNeuralNetForward(
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
	// 		"CudaForwardXor_Convergence",
	// 		EndianString
	// 	);
	// }
	// // SECTION STOP: forward XOR with close to perfect initial weights

	// // SECTION START: MNIST with MSE
	// printf("Starting MNIST training\n");
	// {
	// 	uint32_t MiniBatchSize = 32;
	// 	uint32_t TrainingSamples = 2048;
	// 	uint32_t TestSamples = 100;
	// 	uint32_t Epochs = 100;
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
	// 	CudaAllocMatrix(&Data, TrainingSamples, MNIST_DATA_SIZE);
	// 	MatrixClear(Data);
	// 	CudaAllocMatrix(&Labels, TrainingSamples, MNIST_CLASS_COUNT);
	// 	MatrixClear(Labels);
	// 	int Result = LoadMnistDigitCsv(
	// 		Data, Labels, TrainingSamples, FilePathBuffer
	// 	);

	// 	if(Result == 0)
	// 	{
	// 		neural_net* NeuralNet = NULL;
	// 		CudaAllocNeuralNet(&NeuralNet, MiniBatchSize, MNIST_DATA_SIZE);
	// 		uint32_t HiddenDim = 64;
	// 		CudaAddDense(NeuralNet, HiddenDim);
	// 		CudaAddRelu(NeuralNet);
	// 		CudaAddDense(NeuralNet, HiddenDim);
	// 		CudaAddRelu(NeuralNet);
	// 		CudaAddDense(NeuralNet, MNIST_CLASS_COUNT);

	// 		neural_net_trainer* Trainer;
	// 		CudaAllocNeuralNetTrainer(
	// 			&Trainer,
	// 			NeuralNet,
	// 			LearningRate,
	// 			LayerType_Mse,
	// 			MiniBatchSize,
	// 			Labels->NumColumns
	// 		);

	// 		neural_net* FullBatchNnViewer = NULL;
	// 		CudaResizedNeuralNet(&FullBatchNnViewer, NeuralNet, TrainingSamples);
	// 		neural_net* TestNnViewer = NULL;
	// 		CudaResizedNeuralNet(&TestNnViewer, NeuralNet, TestSamples);

	// 		CudaTrainNeuralNetMiniBatch(
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

	// 		float TrainingAccuracy = CudaTopOneAccuracy(
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

	// 		matrix* TestData = NULL;
	// 		matrix* TestLabels = NULL;
	// 		CudaAllocMatrix(&TestData, TestSamples, MNIST_DATA_SIZE);
	// 		CudaAllocMatrix(&TestLabels, TestSamples, MNIST_CLASS_COUNT);
	// 		Result = LoadMnistDigitCsv(
	// 			TestData, TestLabels, TestSamples, FilePathBuffer
	// 		);
	// 		float TestAccuracy = CudaTopOneAccuracy(
	// 			TestNnViewer, TestData, TestLabels
	// 		);
	// 		printf("TestAccuracy = %f\n", TestAccuracy);

	// 		if(TestAccuracy < 0.9f)
	// 		{
	// 			printf("MNIST training test failed\n");
	// 		}

	// 		// SECTION START: test model saving and loading
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

	// 		// SECTION STOP: test model saving and loading

	// 		// SECTION START: test freeing neural nets
	// 		// TODO: add a check for available memory before and after
	// 		// FreeNeuralNetTrainer(Trainer);
	// 		// FreeNeuralNet(NeuralNet);
	// 		// FreeNeuralNet(LoadedNeuralNet);
	// 		// FreeResizedNeuralNet(FullBatchNnViewer);
	// 		// FreeResizedNeuralNet(TestNnViewer);
	// 		// SECTION STOP: test freeing neural nets
	// 	}
	// 	else
	// 	{
	// 		printf("Unable to run mnist test\n");
	// 	}
	// }
	// // SECTION STOP: MNIST with MSE

	return 0;
}