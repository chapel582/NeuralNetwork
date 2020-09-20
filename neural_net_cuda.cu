// TODO: handle cudaMallocManaged failures
// TODO: query max block size
#include "arg_max.h"
#include "int_shuffler.h"
#include "neural_net.h"
#include "matrix.h"
#include "neural_net_cpu.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

	// TODO: the value returned seems to be either wrong or pointless
	cudaDeviceGetAttribute(
		&GlobalMaxGridDimArray[Device], cudaDevAttrMaxGridDimX, Device
	);
	GlobalMaxGridDimArray[Device] = 64;
}

uint32_t GetBlockSize(uint32_t Device)
{
	return GlobalMaxBlockSizeArray[Device];
}

uint32_t GetMaxNumBlocks(uint32_t Device)
{
	return GlobalMaxGridDimArray[Device];
}

uint32_t GetNumBlocks(uint32_t Range, uint32_t BlockSize, uint32_t Device)
{
	// NOTE: for getting max blocks for operations that are parallelizable 
	// CONT: without no sync

	// NOTE: NumBlocks is always at least one, and grows as the data to 
	// CONT: process grows
	
	uint32_t NumBlocks = (Range + BlockSize - 1) / BlockSize;
	int MaxBlocks = GetMaxNumBlocks(Device);
	if(MaxBlocks < NumBlocks)
	{
		MaxBlocks = NumBlocks;
	}
	return NumBlocks;
}

__global__
void CudaMatrixMultThread(matrix* M1, matrix* M2, matrix* Result)
{	
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	MatrixMultCore(M1, M2, Result, Start, Stride);
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

	AddVectorToRowsCore(M1, Vector, Result, Start, Stride);
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

__global__
void CudaMatrixAddThread(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	MatrixAddCore(M1, M2, Result, Start, Stride);
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

__global__
void CudaMatrixMultM1TransposeThread(matrix* M1, matrix* M2, matrix* Result)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;
	
	MatrixMultM1TransposeCore(M1, M2, Result, Start, Stride);
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

__global__
void CudaMatrixMultM2TransposeThread(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	MatrixMultM2TransposeCore(M1, M2, Result, Start, Stride);
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

__global__
void CudaMatrixMultM1M2TransposeThread(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	MatrixMultM1M2TransposeCore(M1, M2, Result, Start, Stride);
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

__global__
void CudaMatrixScalarMultThread(float Scalar, matrix* M1, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	MatrixScalarMultCore(Scalar, M1, Result, Start, Stride);
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

__global__
void CudaMatrixSubtractThread(matrix* M1, matrix* M2, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	MatrixSubtractCore(M1, M2, Result, Start, Stride);
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

__global__
void CudaMatrixMeanThread(matrix* M1, matrix* Result)
{
	// NOTE: this basically indexes by the thread index, but b/c the thread 
	// CONT: index is reset on every block, 
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	uint32_t Stride = blockDim.x * gridDim.x;

	MatrixMeanCore(M1, Result, Start, Stride);
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
	MatrixMultCore(Inputs, &DenseLayer->Weights, Results, Start, Stride);
	__syncthreads();
	AddVectorToRowsCore(
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
		GetMatrixArrayCount(Results), BlockSize, Device
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
	// NOTE: all of these operations don't have any dependencies on other 
	// CONT: thread's outcomes, so we don't need calls to __syncthreads 

	matrix* Weights = &DenseLayer->Weights;

	// NOTE: Calculate this layer's gradient
	MatrixMultM2TransposeCore(
		NextLayerGradient,
		Weights,
		&TrainData->LayerGradient,
		Start,
		Stride
	);
	__syncthreads();

	// NOTE: Calculate the delta for the weights
	matrix* WeightsDelta = &TrainData->WeightsDelta;
	MatrixMultM1TransposeCore(
		Inputs, NextLayerGradient, WeightsDelta, Start, Stride
	);
	__syncthreads();
	MatrixScalarMultCore(
		TrainData->LearningRate, WeightsDelta, WeightsDelta, Start, Stride
	);
	__syncthreads();
	
	// NOTE: update weights
	MatrixAddCore(Weights, WeightsDelta, Weights, Start, Stride);
	__syncthreads();

	// NOTE: calculate bias delta
	matrix* Bias = &DenseLayer->Bias;
	matrix* BiasDelta = &TrainData->BiasDelta;
	MatrixMeanCore(NextLayerGradient, BiasDelta, Start, Stride);
	__syncthreads();
	MatrixScalarMultCore(
		TrainData->LearningRate, BiasDelta, BiasDelta, Start, Stride
	);
	__syncthreads();
	
	// NOTE: update bias
	MatrixAddCore(Bias, BiasDelta, Bias, Start, Stride);
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
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(&DenseLayer->Weights), BlockSize, Device
	);
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

void CudaFreeReluTrain(relu_train_data* TrainData)
{
	CudaFreeMatrixData(&TrainData->LayerGradient);
	free(TrainData);
}

__global__
void CudaReluForwardThread(matrix* Inputs, matrix* Outputs)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;  
	uint32_t Stride = blockDim.x * gridDim.x;

	ReluForwardCore(Inputs, Outputs, Start, Stride);
}

void CudaReluForward(matrix* Inputs, matrix* Outputs)
{
	assert(Inputs->NumRows == Outputs->NumRows);
	assert(Inputs->NumColumns == Outputs->NumColumns);

	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(Inputs), BlockSize, Device
	);
	CudaReluForwardThread<<<NumBlocks, BlockSize>>>(Inputs, Outputs);
	cudaDeviceSynchronize();
}

__global__
void CudaReluBackThread(
	matrix* Inputs, matrix* NextLayerGradient, matrix* LayerGradient
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = gridDim.x * blockDim.x;
	ReluBackCore(
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
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(Inputs), BlockSize, Device
	);
	CudaReluBackThread<<<NumBlocks, BlockSize>>>(
		Inputs, NextLayerGradient, &TrainData->LayerGradient
	);
	cudaDeviceSynchronize();
}

__device__
float CudaMseForwardCore(
	matrix* Predictions,
	matrix* Labels,
	float* Results,
	uint32_t ResultsCount,
	uint32_t Start,
	uint32_t Stride
)
{
	float ThreadResult = 0.0f;
	if(Start < ResultsCount)
	{
		ThreadResult = MseForwardCore(Predictions, Labels, Start, ResultsCount);
	}
	__syncthreads();
	
	if(Start >= ResultsCount)
	{
		// NOTE: no guarantee we'll have <= threads than results entries
		// NOTE: can't return earlier b/c of the syncthreads call
		return 0.0f;
	}
	else
	{
		Results[Start] = ThreadResult;
	}

	// NOTE: single-threaded summation
	// TODO: could try a divide-and conquer algorithm for fast summation
	float Result = 0.0f;	
	if(Start == 0)
	{
		for(int Index = 0; Index < ResultsCount; Index++)
		{
			Result += Results[Index];
		}

		Result /= (2.0f * Predictions->NumRows);
	}
	return Result;
}

__global__
void CudaMseForwardThread(
	matrix* Predictions,
	matrix* Labels,
	float* GlobalResult,
	uint32_t ResultsCount
)
{
	extern __shared__ float Results[];

	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = gridDim.x * blockDim.x;

	float Result = CudaMseForwardCore(
		Predictions, Labels, Results, ResultsCount, Start, Stride
	);
	if(Start == 0)
	{
		*GlobalResult = Result;
	}
}

float CudaMseForward(matrix* Predictions, matrix* Labels)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(Predictions), BlockSize, Device
	);
	float* Mse;
	cudaMallocManaged(&Mse, sizeof(float));
	uint32_t ResultsCount = NumBlocks * BlockSize;
	size_t MemorySize = sizeof(float) * ResultsCount;
	CudaMseForwardThread<<<NumBlocks, BlockSize, MemorySize>>>(
		Predictions, Labels, Mse, ResultsCount
	);
	cudaDeviceSynchronize();
	float Result = *Mse;
	cudaFree(Mse);
	return Result;
}

__device__
void CudaMseBackCore(
	matrix* Predictions,
	matrix* Labels,
	mse_train_data* TrainData,
	uint32_t Start,
	uint32_t Stride
)
{
	MatrixSubtractCore(
		Labels, Predictions, &TrainData->LayerGradient, Start, Stride
	);
	__syncthreads();
	MatrixScalarMultCore(
		1.0f / Predictions->NumColumns,
		&TrainData->LayerGradient,
		&TrainData->LayerGradient,
		Start,
		Stride
	);
}

__global__
void CudaMseBackThread(
	matrix* Predictions,
	matrix* Labels,
	mse_train_data* TrainData
)
{
	uint32_t Start = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t Stride = gridDim.x * blockDim.x;

	CudaMseBackCore(Predictions, Labels, TrainData, Start, Stride);
}

void CudaMseBack(
	matrix* Predictions,
	matrix* Labels,
	mse_train_data* TrainData
)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetNumBlocks(
		GetMatrixArrayCount(&TrainData->LayerGradient), BlockSize, Device
	);
	CudaMseBackThread<<<NumBlocks, BlockSize>>>(
		Predictions, Labels, TrainData
	);
	cudaDeviceSynchronize();
}

void CudaAllocNeuralNet(
	neural_net** Result,
	uint32_t BatchSize,
	uint32_t InputDim,
	uint32_t CpuThreads
)
{
	cudaMallocManaged(Result, sizeof(neural_net));
	neural_net* NeuralNet = *Result;
	*NeuralNet = {};
	NeuralNet->BatchSize = BatchSize;
	NeuralNet->InputDim = InputDim;
	// TODO: delete this op job allocation
	AllocMatrixOpJobs((matrix_op_jobs**) &NeuralNet->MatrixOpJobs, CpuThreads);
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

void CudaFreeLayerLink(layer_link* LayerLink)
{
	if(LayerLink->Output != NULL)
	{
		CudaFreeMatrix(LayerLink->Output);
	}
	cudaFree(LayerLink);
}

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

void CudaAddMeanSquared(neural_net* NeuralNet)
{
	CudaAddLayerLink(NeuralNet, LayerType_Mse);
}

void CudaFreeNeuralNet(neural_net* NeuralNet)
{
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
				dense_layer* DenseLayer = (dense_layer*) LayerLink->Data;
				CudaFreeDenseLayer(DenseLayer);
				break;
			}
			case(LayerType_Relu):
			{				
				break;
			}
			case(LayerType_Softmax):
			{
				// TODO: NOT IMPLEMENTED
				break;
			}
			case(LayerType_CrossEntropy):
			{
				// TODO: NOT IMPLEMENTED
				break;
			}
			case(LayerType_Mse):
			{
				break;
			}
			default:
			{				
				break;
			}
		}

		layer_link* Next = LayerLink->Next;
		CudaFreeLayerLink(LayerLink);
		LayerLink = Next;
	}
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

	matrix_op_jobs* MatrixOpJobs = Source->MatrixOpJobs;
	CudaAllocNeuralNet(
		Result, NewBatchSize, Source->InputDim, MatrixOpJobs->NumThreads
	);
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
				CudaAddMeanSquared(NeuralNet);
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

void CudaAllocMseTrainData(
	mse_train_data** Result, uint32_t BatchSize, uint32_t PredictionDim
)
{
	cudaMallocManaged(Result, sizeof(mse_train_data));
	mse_train_data* TrainData = *Result;
	*TrainData = {};
	CudaInitMatrix(&TrainData->LayerGradient, BatchSize, PredictionDim);
}

void CudaFreeMseTrainData(mse_train_data* TrainData)
{
	CudaFreeMatrixData(&TrainData->LayerGradient);
	cudaFree(TrainData);
}

void CudaFreeResizedNeuralNet(neural_net* NeuralNet)
{
	layer_link* LayerLink = NeuralNet->FirstLink;
	for(
		uint32_t LayerIndex = 0;
		LayerIndex < NeuralNet->NumLayers;
		LayerIndex++
	)
	{
		layer_link* Next = LayerLink->Next;
		CudaFreeLayerLink(LayerLink);
		LayerLink = Next;
	}
}

__device__
float CudaNeuralNetForwardCore(
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	float* MseResults,
	uint32_t MseResultsCount,
	uint32_t ThreadIndex,
	uint32_t ThreadCount
)
{
	matrix* Outputs = NULL;
	layer_link* LayerLink = NeuralNet->FirstLink;
	float Loss = -1.0f;
	uint32_t NumLayers = NeuralNet->NumLayers;
	for(
		uint32_t LayerIndex = 0;
		LayerIndex < NumLayers;
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
					ThreadIndex,
					ThreadCount
				);
				break;
			}
			case(LayerType_Relu):
			{
				ReluForwardCore(Inputs, Outputs, ThreadIndex, ThreadCount);
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
				if(Labels != NULL)
				{
					Loss = CudaMseForwardCore(
						Inputs,
						Labels,
						MseResults,
						MseResultsCount,
						ThreadIndex,
						ThreadCount
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
		__syncthreads();
	}

	return Loss;
}

__global__
void CudaNeuralNetForwardThread(
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	float* LossResult,
	uint32_t MseResultsCount
)
{
	extern __shared__ float MseResults[];

	uint32_t ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t NumThreads = gridDim.x * blockDim.x;
	float Loss = CudaNeuralNetForwardCore(
		NeuralNet,
		Inputs,
		Labels,
		MseResults,
		MseResultsCount,
		ThreadIndex,
		NumThreads
	);
	__syncthreads();
	if(ThreadIndex == 0 && LossResult)
	{
		*LossResult = Loss;
	}
}

cudaError_t CudaNeuralNetForward(
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	matrix** Predictions,
	float* LossResult
)
{
	int Device = 0;
	uint32_t BlockSize = GetBlockSize(Device);
	uint32_t NumBlocks = GetMaxNumBlocks(Device);
	
	uint32_t MseResultsCount; 
	if(Labels)
	{
		MseResultsCount = Labels->NumRows;	
	}
	else
	{
		MseResultsCount = 1;
	}
	size_t MemorySize = MseResultsCount * sizeof(float);
	CudaNeuralNetForwardThread<<<NumBlocks, BlockSize, MemorySize>>>(
		NeuralNet, Inputs, Labels, LossResult, MseResultsCount
	);
	cudaError_t Error = cudaDeviceSynchronize();

	if(Predictions)
	{
		*Predictions = GetOutput(NeuralNet);
	}
	return Error;
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
			CudaAddMeanSquared(NeuralNet);
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

void CudaFreeNeuralNetTrainer(neural_net_trainer* Trainer)
{
	// NOTE: trainers should be freed before their NNs
	neural_net* NeuralNet = Trainer->NeuralNet;
	void** TrainDataArray = Trainer->TrainDataArray;
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
				CudaFreeDenseLayerTrain(
					(dense_layer_train_data*) TrainDataArray[LayerIndex]					
				);
				break;
			}
			case(LayerType_Relu):
			{
				CudaFreeReluTrain(
					(relu_train_data*) TrainDataArray[LayerIndex]
				);
				break;
			}
			case(LayerType_Softmax):
			{
				// TODO: implement
				break;
			}
			case(LayerType_CrossEntropy):
			{
				// TODO: implement
				break;
			}
			case(LayerType_Mse):
			{
				CudaFreeMseTrainData(
					(mse_train_data*) TrainDataArray[LayerIndex]
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

	cudaFree(TrainDataArray);
	cudaFree(Trainer);
}

void CudaTrainNeuralNet(
	neural_net_trainer* Trainer,
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	uint32_t Epochs,
	bool ShouldInitDenseLayers = true,
	bool PrintStatus = false,
	float* Loss = NULL
)
{
	if(ShouldInitDenseLayers)
	{
		InitDenseLayers(NeuralNet);
	}

	layer_link* LayerLink;

	bool FreeAtEnd = false;
	if(Loss == NULL)
	{
		cudaMallocManaged(&Loss, sizeof(float));
	}

	for(uint32_t Epoch = 0; Epoch < Epochs; Epoch++)
	{
		matrix* Predictions = NULL;
		CudaNeuralNetForward(
			NeuralNet,
			Inputs,
			Labels,
			&Predictions,
			Loss
		);
		if(PrintStatus)
		{
			printf("Epoch %d Loss: %f\n", Epoch, *Loss);
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
					break;
				}
				case(LayerType_Mse):
				{
					mse_train_data* MseTrain = (mse_train_data*) TrainData;

					CudaMseBack(
						Predictions,
						Labels,
						MseTrain
					);
					NextLayerGradient = &MseTrain->LayerGradient;
					break;
				}
				case(LayerType_CrossEntropy):
				{					
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

	if(FreeAtEnd)
	{
		cudaFree(Loss);
	}
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
	neural_net* FullBatchNnViewer = NULL,
	float* Loss = NULL
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
				false,
				Loss
			);
		}

		NeuralNetForward(
			FullBatchNnViewer,
			Inputs,
			Labels,
			NULL,
			Loss
		);
		float TrainingAccuracy = TopOneAccuracy(
			FullBatchNnViewer, Inputs, Labels
		);
		if(PrintStatus)
		{
			printf(
				"Epoch %d Loss, Accuracy: %f, %f\n",
				Epoch,
				*Loss,
				TrainingAccuracy
			);
		}
		if(
			TrainingAccuracy >= TrainingAccuracyThreshold ||
			*Loss <= LossThreshold
		)
		{
			break;
		}
	}
}