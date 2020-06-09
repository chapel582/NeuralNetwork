#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// TODO: Need to have a platform independent way of handling threads
#include <windows.h>

#include "neural_net_common.h"

extern "C" int TempTest()
{
	return 15;
}

void AllocVector(uint64_t Length, vector** Result)
{
	*Result = (vector*) malloc(sizeof(vector));
	vector* Vector = *Result;
	Vector->Length = Length;
	Vector->Data = (float*) malloc(GetVectorDataSize(*Vector));
}

void FreeVector(vector* Vector)
{
	free(Vector->Data);
	free(Vector);
}

void AllocVectorArray(
	uint64_t NumVectors, uint64_t VectorLength, vector_array** Result
)
{
	// NOTE: not in common b/c it's in CPU memory, not shared memory with GPU
	*Result = (vector_array*) malloc(sizeof(vector_array));
	vector_array* VectorArray = *Result;
	VectorArray->Length = NumVectors;
	VectorArray->VectorLength = VectorLength;
	VectorArray->Vectors = (float*) malloc(
		GetVectorArrayDataSize(*VectorArray)
	);
}

void FreeVectorArray(vector_array* VectorArray)
{
	free(VectorArray->Vectors);
	free(VectorArray);
}

void MakeVectorArray(
	uint64_t NumVectors, uint64_t VectorLength, vector_array** Result
)
{
	AllocVectorArray(NumVectors, VectorLength, Result);
	vector_array* VectorArray = *Result;
	memset(VectorArray->Vectors, 0, GetVectorArrayDataSize(*VectorArray));
}

inline void AllocVectorArraySameDim(
	vector_array CopyDimFrom, vector_array** Result
)
{
	return AllocVectorArray(
		CopyDimFrom.Length, CopyDimFrom.VectorLength, Result
	);
}

void AllocLayerOutput(
	uint64_t NumInputs, dense_layer Layer, vector_array** Result
)
{
	AllocVectorArray(NumInputs, Layer.Biases->Length, Result);
}

// TODO: finish CPU threading for matrix operations
// TODO: figure out good error handling and thread-safe assertion scheme 
void AllocDenseLayer(
	uint64_t InputDim, uint64_t OutputDim, dense_layer** Result
)
{
	*Result = (dense_layer*) malloc(sizeof(dense_layer));
	dense_layer* DenseLayer = *Result;	
	
	AllocVectorArray(OutputDim, InputDim, &DenseLayer->Weights);
	
	AllocVector(OutputDim, &DenseLayer->Biases);
}

void MakeDenseLayer(
	uint64_t InputDim, uint64_t OutputDim, dense_layer** Result
)
{
	AllocDenseLayer(InputDim, OutputDim, Result);
	InitDenseLayer(*Result);
}

void FreeDenseLayer(dense_layer* DenseLayer)
{
	FreeVector(DenseLayer->Biases);
	FreeVectorArray(DenseLayer->Weights);
	free(DenseLayer);
}

struct thread_dense_forward_args
{
	vector_array Inputs;
	dense_layer DenseLayer;
	vector_array* Outputs;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadDenseForward(void* VoidArgs)
{
	thread_dense_forward_args* Args = (thread_dense_forward_args*) VoidArgs;
	vector_array Inputs = Args->Inputs;
	dense_layer DenseLayer = Args->DenseLayer;
	vector_array* Outputs = Args->Outputs;
	weights* Weights = DenseLayer.Weights;
	vector* Biases = DenseLayer.Biases;
	float* BiasData = Biases->Data;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Inputs.Length; Row += Stride)
	{
		// NOTE: we might be able to save on a copy here
		float* Input = GetVector(Inputs, Row);
		float* Output = GetVector(*Outputs, Row);
		for(int WeightIndex = 0; WeightIndex < Weights->Length; WeightIndex++)
		{
			float DotResult = 0.0;
			float* WeightVector = GetVector(*Weights, WeightIndex);
			for(
				int ElementIndex = 0;
				ElementIndex < Weights->VectorLength;
				ElementIndex++
			)
			{
				DotResult += (
					Input[ElementIndex] * WeightVector[ElementIndex]
				);
			}
			Output[WeightIndex] = DotResult + BiasData[WeightIndex]; 
		}
	}
	return 0;
}

void DenseForward(
	vector_array Inputs,
	dense_layer DenseLayer,
	vector_array* Outputs,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_dense_forward_args* ThreadArgs
)
{
	// NOTE: only works for windows right now
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_dense_forward_args* Args = &ThreadArgs[Index];
		Args->Inputs = Inputs;
		Args->DenseLayer = DenseLayer;
		Args->Outputs = Outputs;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadDenseForward, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
}

struct thread_relu_forward_args
{
	vector_array Inputs;
	vector_array* Outputs;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadReluForward(void* VoidArgs)
{
	thread_relu_forward_args* Args = (thread_relu_forward_args*) VoidArgs;
	vector_array Inputs = Args->Inputs;
	vector_array* Outputs = Args->Outputs;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Inputs.Length; Row += Stride)
	{
		float* Input = GetVector(Inputs, Row);
		float* Output = GetVector(*Outputs, Row);
		for(
			int ElementIndex = 0;
			ElementIndex < Inputs.VectorLength;
			ElementIndex++
		)
		{
			if(Input[ElementIndex] < 0)
			{
				Output[ElementIndex] = 0;
			}
			else
			{
				Output[ElementIndex] = Input[ElementIndex];
			}
		}
	}

	return 0;
}

void ReluForward(
	vector_array Inputs,
	vector_array* Outputs,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_relu_forward_args* ThreadArgs
)
{
	// NOTE: only works for windows right now
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_relu_forward_args* Args = &ThreadArgs[Index];
		Args->Inputs = Inputs;
		Args->Outputs = Outputs;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadReluForward, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
}

struct thread_sigmoid_forward_args
{
	vector_array Inputs;
	vector_array* Outputs;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadSigmoidForward(void* VoidArgs)
{
	thread_sigmoid_forward_args* Args = (thread_sigmoid_forward_args*) VoidArgs;
	vector_array Inputs = Args->Inputs;
	vector_array* Outputs = Args->Outputs;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Inputs.Length; Row += Stride)
	{
		float* Input = GetVector(Inputs, Row);
		float* Output = GetVector(*Outputs, Row);
		for(
			int ElementIndex = 0;
			ElementIndex < Inputs.VectorLength;
			ElementIndex++
		)
		{
			Output[ElementIndex] = (float) (
				1.0f / (1 + exp(-1 * Input[ElementIndex]))
			);
		}
	}

	return 0;
}

void SigmoidForward(
	vector_array Inputs,
	vector_array* Outputs,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_sigmoid_forward_args* ThreadArgs
)
{
	// NOTE: only works for windows right now
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_sigmoid_forward_args* Args = &ThreadArgs[Index];
		Args->Inputs = Inputs;
		Args->Outputs = Outputs;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadSigmoidForward, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
}

struct thread_softmax_forward_args
{
	vector_array Inputs;
	vector_array* Outputs;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadSoftmaxForward(void* VoidArgs)
{
	thread_softmax_forward_args* Args = (thread_softmax_forward_args*) VoidArgs;
	vector_array Inputs = Args->Inputs;
	vector_array* Outputs = Args->Outputs;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Inputs.Length; Row += Stride)
	{
		float Sum = 0;
		float* Input = GetVector(Inputs, Row);
		float* Output = GetVector(*Outputs, Row);
		for(
			int ElementIndex = 0;
			ElementIndex < Inputs.VectorLength;
			ElementIndex++
		)
		{
			Input[ElementIndex] = (float) exp(Input[ElementIndex]);
			Sum += Input[ElementIndex];
		}

		for(
			int ElementIndex = 0;
			ElementIndex < Inputs.VectorLength;
			ElementIndex++
		)
		{
			Output[ElementIndex] = Input[ElementIndex] / Sum;
		}
	}
	return 0;
}

void SoftmaxForward(
	vector_array Inputs,
	vector_array* Outputs,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_softmax_forward_args* ThreadArgs
)
{
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_softmax_forward_args* Args = &ThreadArgs[Index];
		Args->Inputs = Inputs;
		Args->Outputs = Outputs;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadSoftmaxForward, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
}

struct thread_mse_args
{
	vector_array Predictions;
	vector_array Labels;
	float Result;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadMse(void* VoidArgs)
{
	thread_mse_args* Args = (thread_mse_args*) VoidArgs;
	vector_array Predictions = Args->Predictions;
	vector_array Labels = Args->Labels;
	float* Result = &Args->Result;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Predictions.Length; Row += Stride)
	{
		float* Prediction = GetVector(Predictions, Row);
		float* Label = GetVector(Labels, Row);
		for(
			int ElementIndex = 0;
			ElementIndex < Predictions.VectorLength;
			ElementIndex++
		)
		{
			*Result += (float) pow(
				Label[ElementIndex] - Prediction[ElementIndex], 2
			);
		}
	}
	return 0;
}

float MeanSquaredError(
	vector_array Predictions,
	vector_array Labels,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_mse_args* ThreadArgs
)
{
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_mse_args* Args = &ThreadArgs[Index];
		Args->Labels = Labels;
		Args->Predictions = Predictions;
		Args->Result = 0.0f;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadMse, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
	float Sum = 0;
	for(int Index = 0; Index < NumThreads; Index++)
	{
		thread_mse_args* Args = &ThreadArgs[Index];
		Sum += Args->Result;
	}
	return (Sum / (float) Predictions.Length);
}

struct thread_cross_entropy_args
{
	vector_array Predictions;
	vector_array Labels;
	float Result;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadCrossEntropy(void* VoidArgs)
{
	thread_cross_entropy_args* Args = (thread_cross_entropy_args*) VoidArgs;
	vector_array Predictions = Args->Predictions;
	vector_array Labels = Args->Labels;
	float* Result = &Args->Result;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Predictions.Length; Row += Stride)
	{
		float* Prediction = GetVector(Predictions, Row);
		float* Label = GetVector(Labels, Row);
		for(
			int ElementIndex = 0;
			ElementIndex < Predictions.VectorLength;
			ElementIndex++
		)
		{
			*Result += (float) (
				Label[ElementIndex] * log(Prediction[ElementIndex])
			);
		}
	}
	*Result = -1 * (*Result);
	return 0;
}

float CrossEntropyLoss(
	vector_array Predictions,
	vector_array Labels,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_cross_entropy_args* ThreadArgs
)
{
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_cross_entropy_args* Args = &ThreadArgs[Index];
		Args->Labels = Labels;
		Args->Predictions = Predictions;
		Args->Result = 0.0f;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadCrossEntropy, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
	float Sum = 0;
	for(int Index = 0; Index < NumThreads; Index++)
	{
		thread_cross_entropy_args* Args = &ThreadArgs[Index];
		Sum += Args->Result;
	}
	return Sum;
}

void MakeSpiralData(
	int PointsPerClass,
	int Dimensions,
	int NumClasses,
	vector_array** InputsResults,
	vector_array** OutputsResults
)
{
	// NOTE: this is for testing only
	// NOTE: based on https://cs231n.github.io/neural-networks-case-study/
	// NOTE: data comes normalized already
	AllocVectorArray(NumClasses * PointsPerClass, Dimensions, InputsResults);
	MakeVectorArray(NumClasses * PointsPerClass, NumClasses, OutputsResults);
	InitSpiralData(
		PointsPerClass,
		Dimensions,
		NumClasses,
		*InputsResults,
		*OutputsResults
	);
}