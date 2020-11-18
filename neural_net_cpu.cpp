#include "neural_net.h"

#include "matrix.h"

#include "int_shuffler.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// TODO: Need to have a platform independent way of handling threads
#include <windows.h>

void InitMatrix(matrix* Matrix, uint32_t NumRows, uint32_t NumColumns)
{
	*Matrix = {};
	Matrix->NumRows = NumRows;
	Matrix->NumColumns = NumColumns;
	Matrix->Data = (float*) malloc(GetMatrixDataSize(Matrix));
	memset(Matrix->Data, 0, GetMatrixDataSize(Matrix));
}

void AllocMatrix(matrix** Result, uint32_t NumRows, uint32_t NumColumns)
{
	matrix* Matrix = (matrix*) malloc(sizeof(matrix));
	InitMatrix(Matrix, NumRows, NumColumns);
	*Result = Matrix;
}

void FreeMatrixData(matrix Matrix)
{
	free(Matrix.Data);
}

void FreeMatrix(matrix* Matrix)
{
	FreeMatrixData(*Matrix);
	free(Matrix);
}

void AllocMultResultMatrix(matrix** Result, matrix* M1, matrix* M2)
{
	// NOTE: allocates a matrix that would result from the matrix multiplication
	// CONT: of M1 and M2
	AllocMatrix(Result, M1->NumRows, M2->NumColumns);
}

void AllocM1TransposeMultResultMatrix(matrix** Result, matrix* M1, matrix* M2)
{
	// NOTE: allocates a matrix that would result from the matrix multiplication
	// CONT: of M1 tranposed and M2
	AllocMatrix(Result, M1->NumColumns, M2->NumColumns);
}

void AllocM2TransposeMultResultMatrix(matrix** Result, matrix* M1, matrix* M2)
{
	// NOTE: allocates a matrix that would result from the matrix multiplication
	// CONT: of M1 tranposed and M2
	AllocMatrix(Result, M1->NumRows, M2->NumRows);
}

void AllocM1M2TransposeMultResultMatrix(matrix** Result, matrix* M1, matrix* M2)
{
	// NOTE: allocates a matrix that would result from the matrix multiplication
	// CONT: of M1 tranposed and M2
	AllocMatrix(Result, M1->NumColumns, M2->NumRows);
}

void AllocMatrixMeanResult(matrix** Result, matrix* M1)
{
	AllocMatrix(Result, 1, M1->NumColumns);
}

void AllocMatrixOpJobs(matrix_op_jobs** Result, uint32_t NumThreads)
{
	matrix_op_jobs* Jobs = (matrix_op_jobs*) malloc(sizeof(matrix_op_jobs));
	*Jobs = {};
	Jobs->NumThreads = NumThreads;
	Jobs->Args = (matrix_op_args*) malloc(
		Jobs->NumThreads * sizeof(matrix_op_args)
	);
	memset(Jobs->Args, 0, Jobs->NumThreads * sizeof(matrix_op_args));
	Jobs->Handles = (HANDLE*) malloc(Jobs->NumThreads * sizeof(HANDLE));
	memset(Jobs->Handles, 0, Jobs->NumThreads * sizeof(HANDLE)); 
	*Result = Jobs;
}

void MatrixOpThreadSetupAndRun(
	matrix_op_jobs* MatrixOpJobs,
	matrix* M1,
	matrix* M2,
	matrix* Result,
	LPTHREAD_START_ROUTINE ThreadFunction
)
{
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		matrix_op_args* Args = MatrixOpJobs->Args + ThreadIndex;
		Args->M1 = M1;
		Args->M2 = M2;
		Args->Result = Result;
		Args->Start = ThreadIndex;
		Args->Stride = MatrixOpJobs->NumThreads;
		DWORD ThreadId = 0;
		MatrixOpJobs->Handles[ThreadIndex] = CreateThread(
			NULL, 0, ThreadFunction, Args, 0, &ThreadId 
		);
	}
	WaitForMultipleObjects(
		MatrixOpJobs->NumThreads, MatrixOpJobs->Handles, TRUE, INFINITE
	);
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{		
		CloseHandle(MatrixOpJobs->Handles[ThreadIndex]);
	}
}

DWORD WINAPI CopyMatrixThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	matrix* Destination = Args->M1;
	matrix* Source = Args->M2;
	uint32_t Start = Args->Start;
	uint32_t Stride = Args->Stride;

	for(uint32_t Row = Start; Row < Destination->NumRows; Row += Stride)
	{
		for(uint32_t Column = 0; Column < Destination->NumColumns; Column++)
		{
			SetMatrixElement(
				Destination, Row, Column, GetMatrixElement(Source, Row, Column)
			);
		}
	}
	return 0;
}

void CopyMatrix(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Destination,
	matrix* Source
)
{
	Destination->NumRows = Source->NumRows;
	Destination->NumColumns = Source->NumColumns;
	MatrixOpThreadSetupAndRun(
		MatrixOpJobs, Destination, Source, NULL, CopyMatrixThread
	);
}

DWORD WINAPI MatrixScalarMultThread(void* VoidArgs)
{
	matrix_op_args* Job = (matrix_op_args*) VoidArgs;
	MatrixScalarMultCore(
		Job->Float, Job->M1, Job->Result, Job->Start, Job->Stride
	);
	return 0;
}

void MatrixScalarMult(
	matrix_op_jobs* MatrixOpJobs, float Scalar, matrix* M1, matrix* Result
)
{
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		matrix_op_args* Args = MatrixOpJobs->Args + ThreadIndex;
		Args->Float = Scalar;
		Args->M1 = M1;
		Args->Result = Result;
		Args->Start = ThreadIndex;
		Args->Stride = MatrixOpJobs->NumThreads;
		DWORD ThreadId;
		MatrixOpJobs->Handles[ThreadIndex] = CreateThread(
			NULL, 0, MatrixScalarMultThread, Args, 0, &ThreadId 
		);
	}
	WaitForMultipleObjects(
		MatrixOpJobs->NumThreads, MatrixOpJobs->Handles, TRUE, INFINITE
	);

	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		CloseHandle(MatrixOpJobs->Handles[ThreadIndex]);
	}
}

DWORD WINAPI MatrixMultThread(void* VoidArgs)
{
	matrix_op_args* Job = (matrix_op_args*) VoidArgs;
	MatrixMultCore(Job->M1, Job->M2, Job->Result, Job->Start, Job->Stride);
	return 0;
}

void MatrixMult(
	matrix_op_jobs* MatrixOpJobs, matrix* M1, matrix* M2, matrix* Result
)
{
	assert(M1->NumColumns == M2->NumRows);
	MatrixOpThreadSetupAndRun(MatrixOpJobs, M1, M2, Result, MatrixMultThread);
}

DWORD WINAPI MatrixMultM1TransposeThread(void* VoidArgs)
{
	matrix_op_args* Job = (matrix_op_args*) VoidArgs;
	MatrixMultM1TransposeCore(
		Job->M1, Job->M2, Job->Result, Job->Start, Job->Stride
	);
	return 0;
}

void MatrixMultM1Transpose(
	matrix_op_jobs* MatrixOpJobs, matrix* M1, matrix* M2, matrix* Result
)
{
	assert(M1->NumRows == M2->NumRows);
	MatrixOpThreadSetupAndRun(
		MatrixOpJobs, M1, M2, Result, MatrixMultM1TransposeThread
	);
}

DWORD WINAPI MatrixMultM2TransposeThread(void* VoidArgs)
{
	matrix_op_args* Job = (matrix_op_args*) VoidArgs;
	MatrixMultM2TransposeCore(
		Job->M1, Job->M2, Job->Result, Job->Start, Job->Stride
	);
	return 0;
}

void MatrixMultM2Transpose(
	matrix_op_jobs* MatrixOpJobs, matrix* M1, matrix* M2, matrix* Result
)
{
	// TODO: add assert here
	MatrixOpThreadSetupAndRun(
		MatrixOpJobs, M1, M2, Result, MatrixMultM2TransposeThread
	);
}

DWORD WINAPI MatrixMultM1M2TransposeThread(void* VoidArgs)
{
	matrix_op_args* Job = (matrix_op_args*) VoidArgs;
	MatrixMultM1M2TransposeCore(
		Job->M1, Job->M2, Job->Result, Job->Start, Job->Stride
	);
	return 0;
}

void MatrixMultM1M2Transpose(
	matrix_op_jobs* MatrixOpJobs, matrix* M1, matrix* M2, matrix* Result
)
{
	MatrixOpThreadSetupAndRun(
		MatrixOpJobs, M1, M2, Result, MatrixMultM1M2TransposeThread
	);
}

DWORD WINAPI MatrixAddThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	MatrixAddCore(Args->M1, Args->M2, Args->Result, Args->Start, Args->Stride);
	return 0;
}

void MatrixAdd(
	matrix_op_jobs* MatrixOpJobs, matrix* M1, matrix* M2, matrix* Result
)
{
	assert(M1->NumRows == M2->NumRows);
	assert(M1->NumColumns == M2->NumColumns);
	MatrixOpThreadSetupAndRun(
		MatrixOpJobs, M1, M2, Result, MatrixAddThread
	);
}

DWORD WINAPI MatrixSubtractThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	MatrixSubtractCore(
		Args->M1, Args->M2, Args->Result, Args->Start, Args->Stride
	);
	return 0;
}

void MatrixSubtract(
	matrix_op_jobs* MatrixOpJobs, matrix* M1, matrix* M2, matrix* Result
)
{
	assert(M1->NumRows == M2->NumRows);
	assert(M1->NumColumns == M2->NumColumns);

	MatrixOpThreadSetupAndRun(
		MatrixOpJobs, M1, M2, Result, MatrixSubtractThread
	);
}

DWORD WINAPI AddVectorToRowsThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	AddVectorToRowsCore(
		Args->M1, Args->M2, Args->Result, Args->Start, Args->Stride
	);
	return 0;
}

void AddVectorToRows(
	matrix_op_jobs* MatrixOpJobs, matrix* M1, matrix* Vector, matrix* Result
)
{
	// NOTE: this function is equivalent to adding two matrices, M1 and M2,
	// CONT: where M2 has the same values in each row (Vector) 
	// NOTE: there's no reason to allocate a huge matrix just for this, so this 
	// CONT: method is used instead
	assert(
		(M1->NumColumns == Vector->NumColumns) || 
		(M1->NumColumns == Vector->NumRows)
	);
	MatrixOpThreadSetupAndRun(
		MatrixOpJobs, M1, Vector, Result, AddVectorToRowsThread
	);
}

DWORD WINAPI MatrixMeanThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	MatrixMeanCore(Args->M1, Args->Result, Args->Start, Args->Stride);
	return 0;
}

void MatrixMean(
	matrix_op_jobs* MatrixOpJobs, matrix* M1, matrix* Result
)
{
	/*NOTE:
	This function finds the sum of all the row vectors of matrix M1 and divides
	that sum by the number of rows. 

	M1 Dimensions: N x M
	Result Dimensions: 1 x M
	*/
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		matrix_op_args* Args = MatrixOpJobs->Args + ThreadIndex;
		Args->M1 = M1;
		Args->Result = Result;
		Args->Start = ThreadIndex;
		Args->Stride = MatrixOpJobs->NumThreads;
		DWORD ThreadId;
		MatrixOpJobs->Handles[ThreadIndex] = CreateThread(
			NULL, 0, MatrixMeanThread, Args, 0, &ThreadId 
		);
	}
	WaitForMultipleObjects(
		MatrixOpJobs->NumThreads, MatrixOpJobs->Handles, TRUE, INFINITE
	);
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		CloseHandle(MatrixOpJobs->Handles[ThreadIndex]);
	}
}

void AllocDenseLayer(
	dense_layer** Result, uint32_t InputDim, uint32_t OutputDim
)
{
	dense_layer* DenseLayer = (dense_layer*) malloc(sizeof(dense_layer));
	*DenseLayer = {};
	InitMatrix(&DenseLayer->Weights, InputDim, OutputDim);
	InitMatrix(&DenseLayer->Bias, 1, OutputDim);
	*Result = DenseLayer;
}

void FreeDenseLayer(dense_layer* DenseLayer)
{
	FreeMatrixData(DenseLayer->Weights);
	FreeMatrixData(DenseLayer->Bias);
	free(DenseLayer);
}

void AllocDenseLayerTrain(
	dense_layer_train_data** Result,
	dense_layer* DenseLayer,
	float LearningRate,
	uint32_t BatchSize
)
{
	dense_layer_train_data* TrainData = (dense_layer_train_data*) malloc(
		sizeof(dense_layer_train_data)
	);
	*TrainData = {};
	TrainData->LearningRate = LearningRate; 
	InitMatrix(
		&TrainData->WeightsDelta,
		DenseLayer->Weights.NumRows,
		DenseLayer->Weights.NumColumns
	);
	InitMatrix(
		&TrainData->BiasDelta,
		DenseLayer->Bias.NumRows,
		DenseLayer->Bias.NumColumns
	);
	InitMatrix(
		&TrainData->LayerGradient, BatchSize, DenseLayer->Weights.NumRows
	);
	*Result = TrainData;
}

void FreeDenseLayerTrain(dense_layer_train_data* TrainData)
{
	FreeMatrixData(TrainData->WeightsDelta);
	FreeMatrixData(TrainData->BiasDelta);
	FreeMatrixData(TrainData->LayerGradient);
	free(TrainData);
}

void DenseForward(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Inputs,
	dense_layer* DenseLayer,
	matrix* Results
)
{
	MatrixMult(MatrixOpJobs, Inputs, &DenseLayer->Weights, Results);
	AddVectorToRows(MatrixOpJobs, Results, &DenseLayer->Bias, Results);	
}

void DenseBack(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Inputs,
	matrix* NextLayerGradient,
	dense_layer* DenseLayer,
	dense_layer_train_data* TrainData
)
{
	MatrixMultM2Transpose(
		MatrixOpJobs,
		NextLayerGradient,
		&DenseLayer->Weights,
		&TrainData->LayerGradient
	);

	MatrixMultM1Transpose(
		MatrixOpJobs, Inputs, NextLayerGradient, &TrainData->WeightsDelta
	);
	MatrixScalarMult(
		MatrixOpJobs,
		TrainData->LearningRate,
		&TrainData->WeightsDelta,
		&TrainData->WeightsDelta
	);
	MatrixAdd(
		MatrixOpJobs,
		&DenseLayer->Weights,
		&TrainData->WeightsDelta,
		&DenseLayer->Weights
	);
	
	MatrixMean(MatrixOpJobs, NextLayerGradient, &TrainData->BiasDelta);
	MatrixScalarMult(
		MatrixOpJobs,
		TrainData->LearningRate,
		&TrainData->BiasDelta,
		&TrainData->BiasDelta
	);
	MatrixAdd(
		MatrixOpJobs,
		&DenseLayer->Bias,
		&TrainData->BiasDelta,
		&DenseLayer->Bias
	);
}

void AllocReluTrain(
	relu_train_data** Result, uint32_t BatchSize, uint32_t InputDim
)
{
	relu_train_data* TrainData = (relu_train_data*) malloc(
		sizeof(relu_train_data)
	);
	*TrainData = {};
	InitMatrix(&TrainData->LayerGradient, BatchSize, InputDim);
	*Result = TrainData;
}

void FreeReluTrain(relu_train_data* TrainData)
{
	FreeMatrixData(TrainData->LayerGradient);
	free(TrainData);
}

DWORD WINAPI ReluForwardThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	matrix* M1 = Args->M1;
	matrix* Result = Args->Result;
	uint32_t Start = Args->Start;
	uint32_t Stride = Args->Stride;

	ReluForwardCore(M1, Result, Start, Stride);

	return 0;
}

void ReluForward(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Inputs,
	matrix* Outputs
)
{
	assert(Inputs->NumRows == Outputs->NumRows);
	assert(Inputs->NumColumns == Outputs->NumColumns);

	MatrixOpThreadSetupAndRun(
		MatrixOpJobs, Inputs, NULL, Outputs, ReluForwardThread
	);
}

DWORD WINAPI ReluBackThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	matrix* Inputs = Args->M1;
	matrix* NextLayerGradient = Args->M2;
	matrix* LayerGradient = Args->Result;
	uint32_t Start = Args->Start;
	uint32_t Stride = Args->Stride;
	
	ReluBackCore(
		Inputs,
		NextLayerGradient,
		LayerGradient,
		Start,
		Stride
	);

	return 0;
}

void ReluBack(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Inputs,
	matrix* NextLayerGradient,
	relu_train_data* TrainData
)
{
	matrix* LayerGradient = &TrainData->LayerGradient;
	assert(NextLayerGradient->NumRows == LayerGradient->NumRows);
	assert(NextLayerGradient->NumColumns == LayerGradient->NumColumns);

	MatrixOpThreadSetupAndRun(
		MatrixOpJobs,
		Inputs, 
		NextLayerGradient, 
		LayerGradient,
		ReluBackThread
	);
}

void AllocSoftmaxLayer(
	softmax_layer** Result, uint32_t BatchSize, uint32_t Dim
)
{
	softmax_layer* SoftmaxLayer = (softmax_layer*) malloc(
		sizeof(softmax_layer)
	);
	InitMatrix(&SoftmaxLayer->Intermediate, BatchSize, Dim);
	*Result = SoftmaxLayer;
}

struct softmax_train_data
{
	matrix LayerGradient;
};

void AllocSoftmaxTrain(
	softmax_train_data** Result, uint32_t BatchSize, uint32_t InputDim
)
{
	softmax_train_data* TrainData = (softmax_train_data*) malloc(
		sizeof(softmax_train_data)
	);
	InitMatrix(&TrainData->LayerGradient, BatchSize, InputDim);
	*Result = TrainData;
}

DWORD WINAPI SoftmaxForwardThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	matrix* Inputs = Args->M1;
	matrix* Intermediate = Args->M2;
	matrix* Result = Args->Result;
	
	SoftmaxForwardCore(Inputs, Intermediate, Result, Args->Start, Args->Stride);

	return 0;
}

void SoftmaxForward(
	matrix_op_jobs* MatrixOpJobs,
	softmax_layer* SoftmaxLayer,
	matrix* Inputs,
	matrix* Outputs
)
{
	// NOTE: only used in conjunction with cross-entropy loss right now
	MatrixOpThreadSetupAndRun(
		MatrixOpJobs,
		Inputs,
		&SoftmaxLayer->Intermediate,
		Outputs,
		SoftmaxForwardThread
	);
}

DWORD WINAPI CrossEntropyForwardThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	matrix* Predictions = Args->M1;
	matrix* Labels = Args->M2;
	int Start = Args->Start;
	int Stride = Args->Stride;

	Args->Float = XentropyForwardCore(Predictions, Labels, Start, Stride);
	return 0;
}

float CrossEntropyForward(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Predictions,
	matrix* Labels
)
{
	// NOTE: only used in conjunction with softmax right now
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		matrix_op_args* Args = MatrixOpJobs->Args + ThreadIndex;
		Args->M1 = Predictions;
		Args->M2 = Labels;
		Args->Float = 0.0f;
		Args->Start = ThreadIndex;
		Args->Stride = MatrixOpJobs->NumThreads;
		DWORD ThreadId;
		MatrixOpJobs->Handles[ThreadIndex] = CreateThread(
			NULL, 0, CrossEntropyForwardThread, Args, 0, &ThreadId 
		);
	}
	WaitForMultipleObjects(
		MatrixOpJobs->NumThreads, MatrixOpJobs->Handles, TRUE, INFINITE
	);
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		CloseHandle(MatrixOpJobs->Handles[ThreadIndex]);
	}
	float Sum = 0;
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		matrix_op_args* Args = MatrixOpJobs->Args + ThreadIndex;
		Sum += Args->Float;
	}
	return Sum;
}

void AllocSoftmaxXentropyTrain(
	softmax_xentropy_train_data** Result, softmax_layer* SoftmaxLayer
)
{
	softmax_xentropy_train_data* TrainData = (
		(softmax_xentropy_train_data*) malloc(
			sizeof(softmax_xentropy_train_data)
		)
	);

	InitMatrix(
		&TrainData->LayerGradient,
		SoftmaxLayer->Intermediate.NumRows,
		SoftmaxLayer->Intermediate.NumColumns
	);
	*Result = TrainData;
}

void SoftmaxCrossEntropyBack(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Predictions, 
	matrix* Labels, 
	softmax_xentropy_train_data* TrainData
)
{
	// NOTE: Predictions is the output of softmax layer
	// NOTE: Labels has dim k x m where k is the batch size and m is the # of 
	// CONT: classes
	MatrixSubtract(
		MatrixOpJobs, Labels, Predictions, &TrainData->LayerGradient
	);
	MatrixScalarMult(
		MatrixOpJobs,
		1.0f / Predictions->NumColumns,
		&TrainData->LayerGradient,
		&TrainData->LayerGradient
	);
}

DWORD WINAPI MseForwardThread(void* VoidArgs)
{
	matrix_op_args* Args = (matrix_op_args*) VoidArgs;
	matrix* Predictions = Args->M1;
	matrix* Labels = Args->M2;
	int Start = Args->Start;
	int Stride = Args->Stride;

	Args->Float = MseForwardCore(Predictions, Labels, Start, Stride);
	return 0;
}

float MseForward(
	matrix_op_jobs* MatrixOpJobs, matrix* Predictions, matrix* Labels
)
{
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		matrix_op_args* Args = MatrixOpJobs->Args + ThreadIndex;
		Args->M1 = Predictions;
		Args->M2 = Labels;
		Args->Float = 0.0f;
		Args->Start = ThreadIndex;
		Args->Stride = MatrixOpJobs->NumThreads;
		DWORD ThreadId;
		MatrixOpJobs->Handles[ThreadIndex] = CreateThread(
			NULL, 0, MseForwardThread, Args, 0, &ThreadId 
		);
	}
	WaitForMultipleObjects(
		MatrixOpJobs->NumThreads, MatrixOpJobs->Handles, TRUE, INFINITE
	);
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		CloseHandle(MatrixOpJobs->Handles[ThreadIndex]);
	}
	float Sum = 0;
	for(
		uint32_t ThreadIndex = 0;
		ThreadIndex < MatrixOpJobs->NumThreads;
		ThreadIndex++
	)
	{
		matrix_op_args* Args = MatrixOpJobs->Args + ThreadIndex;
		Sum += Args->Float;
	}
	// NOTE: this definition of MSE with a two in the denominator helps cancel 
	// CONT: out a two in the back derivation 
	float Mean = Sum / (2 * Predictions->NumRows);
	return Mean;
}

void AllocMseTrainData(
	mse_train_data** Result, uint32_t BatchSize, uint32_t PredictionDim
)
{
	mse_train_data* TrainData = (mse_train_data*) malloc(
		sizeof(mse_train_data)
	);
	*TrainData = {};
	InitMatrix(&TrainData->LayerGradient, BatchSize, PredictionDim);
	*Result = TrainData;
}

void FreeMseTrainData(mse_train_data* TrainData)
{
	FreeMatrixData(TrainData->LayerGradient);
	free(TrainData);
}

void MeanSquaredBack(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Predictions, 
	matrix* Labels, 
	mse_train_data* TrainData
)
{
	MatrixSubtract(
		MatrixOpJobs, Labels, Predictions, &TrainData->LayerGradient
	);
	MatrixScalarMult(
		MatrixOpJobs,
		1.0f / Predictions->NumColumns,
		&TrainData->LayerGradient,
		&TrainData->LayerGradient
	);
}

void AllocNeuralNet(
	neural_net** Result,
	uint32_t BatchSize,
	uint32_t InputDim,
	uint32_t NumThreads
)
{
	neural_net* NeuralNet = (neural_net*) malloc(sizeof(neural_net));
	*NeuralNet = {};
	NeuralNet->BatchSize = BatchSize;
	NeuralNet->InputDim = InputDim;
	AllocMatrixOpJobs((matrix_op_jobs**) &NeuralNet->MatrixOpJobs, NumThreads);
	*Result = NeuralNet;
}

uint32_t AddLayerLink(neural_net* NeuralNet, layer_type LayerType)
{
	// TODO: probably should rename this function to no-parameter layer link
	layer_link* LayerLink = (layer_link*) malloc(sizeof(layer_link));
	*LayerLink = {};
	LayerLink->Type = LayerType;
	matrix* Output = (matrix*) NeuralNet->LastLink->Output;
	uint32_t InputDim = Output->NumColumns;
	NeuralNet->LastLink->Next = LayerLink;
	LayerLink->Previous = NeuralNet->LastLink;
	LayerLink->Next = NULL;
	NeuralNet->LastLink = LayerLink;
	NeuralNet->NumLayers++;

	return InputDim;
}

void FreeLayerLink(layer_link* LayerLink)
{
	if(LayerLink->Output != NULL)
	{
		FreeMatrix((matrix*) LayerLink->Output);
	}
	free(LayerLink);
}

void AddDense(
	neural_net* NeuralNet, uint32_t OutputDim, dense_layer* DenseLayer = NULL
)
{
	layer_link* LayerLink = (layer_link*) malloc(sizeof(layer_link));
	*LayerLink = {};
	uint32_t InputDim = 0;
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
		matrix* Output = (matrix*) NeuralNet->LastLink->Output;
		InputDim = Output->NumColumns;
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
		AllocDenseLayer(
			(dense_layer**) &LayerLink->Data, 
			InputDim,
			OutputDim
		);
	}
	AllocMatrix((matrix**) &LayerLink->Output, NeuralNet->BatchSize, OutputDim);

	NeuralNet->NumLayers++;
}

void AddRelu(neural_net* NeuralNet)
{
	uint32_t InputDim = AddLayerLink(NeuralNet, LayerType_Relu);
	layer_link* LayerLink = NeuralNet->LastLink;

	AllocMatrix((matrix**) &LayerLink->Output, NeuralNet->BatchSize, InputDim);
}

void AddSoftmax(neural_net* NeuralNet)
{
	uint32_t InputDim = AddLayerLink(NeuralNet, LayerType_Softmax);
	layer_link* LayerLink = NeuralNet->LastLink;

	AllocSoftmaxLayer(
		(softmax_layer**) &LayerLink->Data, NeuralNet->BatchSize, InputDim
	);
	AllocMatrix((matrix**) &LayerLink->Output, NeuralNet->BatchSize, InputDim);
}

void AddCrossEntropy(neural_net* NeuralNet)
{
	AddLayerLink(NeuralNet, LayerType_CrossEntropy);
}

void AddSoftmaxCrossEntropy(neural_net* NeuralNet)
{
	uint32_t InputDim = AddLayerLink(NeuralNet, LayerType_SoftmaxCrossEntropy);
	layer_link* LayerLink = NeuralNet->LastLink;

	AllocSoftmaxLayer(
		(softmax_layer**) &LayerLink->Data, NeuralNet->BatchSize, InputDim
	);
	AllocMatrix((matrix**) &LayerLink->Output, NeuralNet->BatchSize, InputDim);
}

void AddMeanSquared(neural_net* NeuralNet)
{
	AddLayerLink(NeuralNet, LayerType_Mse);
}

void FreeNeuralNet(neural_net* NeuralNet)
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
				FreeDenseLayer(DenseLayer);
				break;
			}
			case(LayerType_Relu):
			{				
				break;
			}
			case(LayerType_Softmax):
			{
				// TODO: not implemented
				assert(false);
				break;
			}
			case(LayerType_CrossEntropy):
			{
				// TODO: not implemented
				assert(false);
				break;
			}
			case(LayerType_SoftmaxCrossEntropy):
			{
				// TODO: not implemented
				assert(false);
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
		FreeLayerLink(LayerLink);
		LayerLink = Next;
	}
}

void ResizedNeuralNet(
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
	AllocNeuralNet(
		Result,
		NewBatchSize,
		Source->InputDim,
		MatrixOpJobs->NumThreads
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
				AddDense(NeuralNet, DenseLayer->Weights.NumColumns, DenseLayer);
				break;
			}
			case(LayerType_Relu):
			{
				AddRelu(NeuralNet);
				break;
			}
			case(LayerType_Softmax):
			{
				AddSoftmax(NeuralNet);
				break;
			}
			case(LayerType_CrossEntropy):
			{
				AddCrossEntropy(NeuralNet);
				break;
			}
			case(LayerType_SoftmaxCrossEntropy):
			{
				AddSoftmaxCrossEntropy(NeuralNet);
				break;
			}
			case(LayerType_Mse):
			{
				AddMeanSquared(NeuralNet);
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

void FreeResizedNeuralNet(neural_net* NeuralNet)
{
	layer_link* LayerLink = NeuralNet->FirstLink;
	for(
		uint32_t LayerIndex = 0;
		LayerIndex < NeuralNet->NumLayers;
		LayerIndex++
	)
	{
		layer_link* Next = LayerLink->Next;
		FreeLayerLink(LayerLink);
		LayerLink = Next;
	}
}

void NeuralNetForward(
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
	matrix_op_jobs* MatrixOpJobs = (matrix_op_jobs*) NeuralNet->MatrixOpJobs;
	for(
		uint32_t LayerIndex = 0;
		LayerIndex < NeuralNet->NumLayers;
		LayerIndex++
	)
	{
		Outputs = (matrix*) LayerLink->Output;
		switch(LayerLink->Type)
		{
			case(LayerType_Dense):
			{
				DenseForward(
					MatrixOpJobs,
					Inputs,
					(dense_layer*) LayerLink->Data,
					Outputs
				);
				break;
			}
			case(LayerType_Relu):
			{
				ReluForward(MatrixOpJobs, Inputs, Outputs);
				break;
			}
			case(LayerType_Softmax):
			{
				SoftmaxForward(
					MatrixOpJobs,
					(softmax_layer*) LayerLink->Data,
					Inputs,
					Outputs
				);
				break;
			}

			// NOTE: for NNs with loss layers, predictions must be captured 
			// CONT: with inputs the end of the loop since outputs 
			// CONT: will be updated to NULL
			case(LayerType_CrossEntropy):
			{
				// if(Predictions)
				// {
				// 	*Predictions = Inputs;
				// }

				// if(Labels != NULL)
				// {
				// 	Loss = CrossEntropyForward(MatrixOpJobs, Inputs, Labels);
				// }
				// TODO: implement
				assert(false);
				break;
			}
			case(LayerType_SoftmaxCrossEntropy):
			{
				SoftmaxForward(
					MatrixOpJobs,
					(softmax_layer*) LayerLink->Data,
					Inputs,
					Outputs
				);

				if(Predictions)
				{
					*Predictions = Outputs;
				}

				if(Labels != NULL)
				{
					Loss = CrossEntropyForward(MatrixOpJobs, Outputs, Labels);
				}
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
					Loss = MseForward(MatrixOpJobs, Inputs, Labels);
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

int Predict(neural_net* NeuralNet, matrix* Inputs, uint32_t BatchIndex)
{
	// NOTE: utility function for predicting
	matrix* Predictions = NULL;
	NeuralNetForward(
		NeuralNet,
		Inputs,
		NULL,
		&Predictions,
		NULL
	);
	return ArgMax(
		GetMatrixRow(Predictions, BatchIndex), Predictions->NumColumns
	);
}

void AllocNeuralNetTrainer(
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
			AddMeanSquared(NeuralNet);
			break;
		}
		case(LayerType_CrossEntropy):
		{
			AddCrossEntropy(NeuralNet);
			break;
		}
		case(LayerType_SoftmaxCrossEntropy):
		{
			AddSoftmaxCrossEntropy(NeuralNet);
			break;
		}
		default:
		{
			break;
		}
	}

	neural_net_trainer* Trainer = (neural_net_trainer*) (
		malloc(sizeof(neural_net_trainer))
	);
	*Trainer = {};
	Trainer->NeuralNet = NeuralNet;
	void** TrainDataArray = (void**) (
		malloc(NeuralNet->NumLayers * sizeof(void*))
	);
	memset(TrainDataArray, 0, NeuralNet->NumLayers * sizeof(void*));
	Trainer->TrainDataArray = TrainDataArray;

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
				AllocDenseLayerTrain(
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
				matrix* PrevOutputs = (matrix*) PreviousLayer->Output;
				AllocReluTrain(
					(relu_train_data**) &TrainDataArray[LayerIndex],
					NeuralNet->BatchSize,
					PrevOutputs->NumColumns
				);
				break;
			}
			case(LayerType_Softmax):
			{
				// TODO: not implemented
				assert(false);
				break;
			}
			case(LayerType_CrossEntropy):
			{
				// TODO: not implemented
				assert(false);
				break;
			}
			case(LayerType_SoftmaxCrossEntropy):
			{
				// layer_link* PreviousLayer = LayerLink->Previous;
				softmax_layer* SoftmaxLayer = (softmax_layer*)(
					LayerLink->Data
				);

				AllocSoftmaxXentropyTrain(
					(softmax_xentropy_train_data**) &TrainDataArray[LayerIndex],
					SoftmaxLayer
				);
				break;
			}
			case(LayerType_Mse):
			{
				layer_link* PreviousLayer = LayerLink->Previous;
				matrix* PrevOutputs = (matrix*) PreviousLayer->Output;
				AllocMseTrainData(
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

void AllocNeuralNetTrainer(
	neural_net_trainer** Result,
	neural_net* NeuralNet,
	float LearningRate,
	layer_type LossLayer,
	uint32_t MiniBatchSize,
	uint32_t OutputDim
)
{
	// NOTE: function also allocates minibatch matrices
	AllocNeuralNetTrainer(Result, NeuralNet, LearningRate, LossLayer);
	neural_net_trainer* Trainer = *Result;
	AllocMatrix(&Trainer->MiniBatchData, MiniBatchSize, NeuralNet->InputDim);
	AllocMatrix(&Trainer->MiniBatchLabels, MiniBatchSize, OutputDim);
}

void FreeNeuralNetTrainer(neural_net_trainer* Trainer)
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
				FreeDenseLayerTrain(
					(dense_layer_train_data*) TrainDataArray[LayerIndex]					
				);
				break;
			}
			case(LayerType_Relu):
			{
				FreeReluTrain(
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
				FreeMseTrainData(
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

	free(TrainDataArray);
	free(Trainer);
}

void TrainNeuralNet(
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

	matrix_op_jobs* MatrixOpJobs = (matrix_op_jobs*) NeuralNet->MatrixOpJobs;
	layer_link* LayerLink;
	float Loss = -1.0f;
	for(uint32_t Epoch = 0; Epoch < Epochs; Epoch++)
	{
		matrix* Predictions = NULL;
		NeuralNetForward(
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
				LayerInputs = (matrix*) PreviousLayer->Output;
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
					DenseBack(
						MatrixOpJobs,
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
					ReluBack(
						MatrixOpJobs,
						LayerInputs,
						NextLayerGradient,
						ReluTrain
					);
					NextLayerGradient = &ReluTrain->LayerGradient;
					break;
				}
				case(LayerType_Softmax):
				{
					// TODO: not implemented
					assert(false);
					break;
				}
				case(LayerType_CrossEntropy):
				{
					// TODO: not implemented
					assert(false);
					break;
				}
				case(LayerType_SoftmaxCrossEntropy):
				{
					softmax_xentropy_train_data* XEntropyTrain = (
						(softmax_xentropy_train_data*) TrainData
					);

					SoftmaxCrossEntropyBack(
						MatrixOpJobs,
						Predictions, 
						Labels,
						XEntropyTrain
					);
					NextLayerGradient = &XEntropyTrain->LayerGradient;
					break;
				}
				case(LayerType_Mse):
				{
					mse_train_data* MseTrain = (mse_train_data*) TrainData;

					MeanSquaredBack(
						MatrixOpJobs,
						Predictions,
						Labels,
						MseTrain
					);
					NextLayerGradient = &MseTrain->LayerGradient;
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

float TopOneAccuracy(neural_net* NeuralNet, matrix* Inputs, matrix* Labels)
{
	matrix* Predictions = NULL;
	NeuralNetForward(
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

void TrainNeuralNetMiniBatch(
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
			CreateMiniBatch(
				&IntShuffler,
				MiniBatchData,
				MiniBatchLabels,
				Inputs,
				Labels,
				BatchIndex,
				MiniBatchSize,
				0,
				1
			);

			// NOTE: train on mini batch
			TrainNeuralNet(
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
		NeuralNetForward(
			FullBatchNnViewer,
			Inputs,
			Labels,
			NULL,
			&Loss
		);
		float TrainingAccuracy = TopOneAccuracy(
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

	// TODO: free int shuffler
}

void LoadNeuralNet(
	neural_net** Result, char* FilePath, uint32_t BatchSize, uint32_t NumThreads
)
{
	FILE* File;
	fopen_s(&File, FilePath, "rb");
	
	model_header ModelHeader = {};
	fread(&ModelHeader, 1, sizeof(model_header), File);

	AllocNeuralNet(
		Result,
		BatchSize,
		ModelHeader.InputDim,
		NumThreads
	);
	neural_net* NeuralNet = *Result;

	for(
		uint32_t LayerIndex = 0;
		LayerIndex < ModelHeader.NumLayers;
		LayerIndex++
	)
	{
		layer_header LayerHeader = {};
		fread(&LayerHeader, 1, sizeof(layer_header), File);

		switch(LayerHeader.Type)
		{
			case(LayerType_Dense):
			{
				matrix WeightsInfo;
				fread(&WeightsInfo, 1, sizeof(matrix), File);
				matrix BiasInfo;
				fread(&BiasInfo, 1, sizeof(matrix), File);

				AddDense(NeuralNet, WeightsInfo.NumColumns);

				dense_layer* DenseLayer = (dense_layer*)(
					NeuralNet->LastLink->Data
				);
				fread(
					DenseLayer->Weights.Data,
					1,
					GetMatrixDataSize(&DenseLayer->Weights),
					File
				);
				fread(
					DenseLayer->Bias.Data,
					1,
					GetMatrixDataSize(&DenseLayer->Bias),
					File
				);
				break;
			}
			case(LayerType_Relu):
			{
				AddRelu(NeuralNet);
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
				AddMeanSquared(NeuralNet);
				break;
			}
			default:
			{
				break;
			}
		}
	}

	fclose(File);
}

struct conv_layer
{
	uint32_t Stride;
	uint32_t Pad;

	uint32_t InputWidth;
	uint32_t InputHeight;
	uint32_t OutputWidth;
	uint32_t OutputHeight;

	uint32_t ChannelCount;

	// TODO: we assume 2d filters. we might need to handle 3d filters? or even n dimensional filters...?
	uint32_t FilterDim;
	float* Filter;

	float* Bias;
};

float* GetFilter(
	float* Filter,
	uint32_t FilterDim,
	uint32_t ChannelCount,
	uint32_t ChannelIndex
)
{
	assert(ChannelIndex < ChannelCount);
	return Filter + (ChannelIndex * FilterDim * FilterDim);
}

void AllocConv(
	conv_layer** Result,
	uint32_t FilterDim,
	uint32_t Stride,
	uint32_t PrevChannelCount,
	uint32_t ChannelCount
)
{
	*Result = (conv_layer*) malloc(sizeof(conv_layer));
	conv_layer* ConvLayer = *Result;

	*ConvLayer = {};
	ConvLayer->Stride = Stride;
	ConvLayer->ChannelCount = ChannelCount;
	ConvLayer->FilterDim = FilterDim;
	ConvLayer->Filter = (float*) malloc(
		(FilterDim * FilterDim) * 
		ConvLayer->ChannelCount * 
		PrevChannelCount * 
		sizeof(float)
	);
	ConvLayer->Bias = (float*) malloc(ConvLayer->ChannelCount * sizeof(float));
}

void InitConv(conv_layer* ConvLayer, uint32_t InputWidth, uint32_t InputHeight)
{
	ConvLayer->InputWidth = InputWidth;
	ConvLayer->InputHeight = InputHeight;

	ConvLayer->OutputWidth = (
		(
			(InputWidth - ConvLayer->FilterDim + 2 * ConvLayer->Pad) / 
			ConvLayer->Stride
		) + 1
	);
	ConvLayer->OutputHeight = (
		(
			(InputHeight - ConvLayer->FilterDim + 2 * ConvLayer->Pad) / 
			ConvLayer->Stride
		) + 1
	);
}

inline uint32_t ConvOutputSize(conv_layer* ConvLayer)
{
	return (
		ConvLayer->ChannelCount * 
		ConvLayer->OutputWidth * 
		ConvLayer->OutputHeight * 
		sizeof(float)
	);
}

void AllocConvOutput(void** Output, conv_layer* ConvLayer)
{
	*Output = malloc(ConvOutputSize(ConvLayer));
}

// void AddConv(
// 	neural_net* NeuralNet,
// 	uint32_t FilterDim,
// 	uint32_t Stride,
// 	uint32_t Pad,
// 	uint32_t ChannelCount
// )
// {
// 	// NOTE: function is for adding a Conv after a conv layer
// 	layer_link* LayerLink = (layer_link*) malloc(sizeof(layer_link));
// 	*LayerLink = {};
// 	uint32_t PrevChannelCount = 0;
	
// 	LayerLink->Previous = NeuralNet->LastLink;
// 	NeuralNet->LastLink->Next = LayerLink;
// 	NeuralNet->LastLink = LayerLink;
// 	conv_layer* PrevConv = (conv_layer*) NeuralNet->LastLink->Data;
// 	PrevChannelCount = PrevConv->ChannelCount;
// 	uint32_t InputWidth = PrevConv->OutputWidth;
// 	uint32_t InputHeight = PrevConv->OutputHeight;

// 	LayerLink->Type = LayerType_Conv;

// 	AllocConv(
// 		(conv_layer**) &LayerLink->Data,
// 		FilterDim,
// 		Stride,
// 		PrevChannelCount,
// 		ChannelCount
// 	);
// 	InitConv((conv_layer*) LayerLink->Data, InputWidth, InputHeight);
// 	AllocConvOutput(&LayerLink->Output, (conv_layer*) LayerLink->Data);

// 	NeuralNet->NumLayers++;
// }

inline float* GetChannelElement(
	float* Inputs,
	uint32_t InputWidth,
	uint32_t InputHeight,
	uint32_t ChannelIndex,
	uint32_t X,
	uint32_t Y
)
{
	return (
		Inputs + ChannelIndex * InputWidth * InputHeight + Y * InputWidth + X
	);
}

void PrintConvData(
	float* Output, uint32_t ChannelCount, uint32_t Width, uint32_t Height
)
{
	for(
		uint32_t ChannelIndex = 0;
		ChannelIndex < ChannelCount;
		ChannelIndex++
	)
	{
		printf("Channel %d\n", ChannelIndex);
		for(uint32_t Y = 0; Y < Height; Y++)
		{
			printf("[");
			for(uint32_t X = 0; X < Width; X++)
			{
				float* Element = GetChannelElement(
					Output, Width, Height, ChannelIndex, X, Y
				);
				printf("%f, ", *Element);
			}
			printf("]\n");
		}
	}
}

void ZeroPad(
	float* Inputs,
	uint32_t Pad,
	uint32_t ChannelCount,
	uint32_t PostPadWidth,
	uint32_t PostPadHeight
)
{
	for(uint32_t ChannelIndex = 0; ChannelIndex < ChannelCount; ChannelIndex++)
	{
		// NOTE: 0 fill first two rows
		for(uint32_t Y = 0; Y < Pad; Y++)
		{
			for(uint32_t X = 0; X < PostPadWidth; X++)
			{
				float* Element = GetChannelElement(
					Inputs,
					PostPadWidth,
					PostPadHeight,
					ChannelIndex,
					X,
					Y
				);
				*Element = 0.0f;
			}
		}
		// NOTE: 0 fill first two columns
		for(uint32_t Y = 0; Y < PostPadHeight; Y++)
		{
			for(uint32_t X = 0; X < Pad; X++)
			{
				float* Element = GetChannelElement(
					Inputs,
					PostPadWidth,
					PostPadHeight,
					ChannelIndex,
					X,
					Y
				);
				*Element = 0.0f;
			}
		}
	}
}

// void ConvForwardCore(
// 	float* Inputs,
// 	uint32_t PrevChannelCount,
// 	uint32_t BatchSize,
// 	conv_layer* ConvLayer,
// 	uint32_t ThreadIndex,
// 	uint32_t ThreadCount,
// 	float* Output,
// 	uint32_t OutputSize
// )
// {
// 	assert(OutputSize == ConvOutputSize(ConvLayer));

// 	uint32_t XBound = ConvLayer->InputWidth - ConvLayer->Pad;
// 	uint32_t YBound = ConvLayer->InputHeight - ConvLayer->Pad;

// 	// TODO: explore fancier algorithms that take advantage of cache locality
// 	for(
// 		uint32_t InputElementIndex = ThreadIndex;
// 		InputElementIndex < OutputSize;
// 		InputElementIndex += ThreadCount
// 	)
// 	{
// 		// float* ResultElement = Output + ResultElementIndex;
// 		// TODO: double check these x and y calculations
// 		// uint32_t ChannelElementCount = (
// 		// 	ConvLayer->OutputWidth * ConvLayer->OutputHeight
// 		// );
// 		// uint32_t ResultChannelIndex = ResultElementIndex / ChannelElementCount;
// 		// uint32_t OffsetResultElementIndex = (
// 		// 	ResultElementIndex % ChannelElementCount
// 		// );
// 		// uint32_t ResultY = OffsetResultElementIndex / ConvLayer->OutputWidth;
// 		// uint32_t ResultX = OffsetResultElementIndex % ConvLayer->OutputWidth;
// 		// for(
// 		// 	uint32_t X = 0; X < ConvLayer->FilterDim; X++
// 		// )
// 		// {
// 		// 	for(uint32_t Y = 0; Y < ConvLayer->FilterDim; Y++)
// 		// 	{
// 		// 		GetElementFromConv(
// 		// 			Inputs, ResultChannelIndex, ResultX + X, ResultY + Y
// 		// 		);
// 		// 	}
// 		// }
		
// 		// TODO: need to include batch size in these calculations
// 		uint32_t ChannelElementCount = (
// 			ConvLayer->OutputWidth * ConvLayer->OutputHeight
// 		);
// 		uint32_t InputChannelIndex = InputElementIndex / ChannelElementCount;
// 		uint32_t OffsetInputElementIndex = (
// 			InputElementIndex % ChannelElementCount
// 		);
// 		uint32_t InputY = OffsetInputElementIndex / ConvLayer->OutputWidth;
// 		uint32_t InputX = OffsetInputElementIndex % ConvLayer->OutputWidth;
// 		if(InputX >= XBound || InputY >= YBound)
// 		{
// 			continue;
// 		}

// 		float Sum = 0.0f;
// 		for(uint32_t X = 0; X < ConvLayer->FilterDim; X++)
// 		{
// 			for(uint32_t Y = 0; Y < ConvLayer->FilterDim; Y++)
// 			{
// 				float FilterElement = GetFilterElement(ConvLayer, OutputChannelIndex, X, Y);
// 				float InputElement = GetChannelElement(
// 					Inputs,
// 					PrevChannelCount,
// 					ConvLayer->InputWidth,
// 					ConvLayer->InputHeight,
// 					InputChannelIndex,
// 					InputX + X,
// 					InputY + Y
// 				);
// 				Sum += FilterElement * InputElement;
// 			}
// 		}
// 	}
// }