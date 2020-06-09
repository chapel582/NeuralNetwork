#include <stdio.h>
#include <windows.h>

typedef int temp_test();

int main(void)
{
	HMODULE Module = LoadLibraryA("neural_net_cpu.dll");
	if(Module == NULL)
	{
		goto error;
	}
	// temp_test* TempTest = (temp_test*) GetProcAddress(Module, "TempTest");
	// DWORD LastError = GetLastError();
	// printf("%d", LastError);
	// if(TempTest == NULL)
	// {
	// 	goto error;
	// }
	// printf("%d", TempTest());
	
	/*
	TODO: remove this silly temp data result reminder
	[4.800000, 1.210000, 2.385000]
	[8.900000, -1.810000, 0.200000]

	[4.800000, -1.210000, 1.192500]
	[8.900000, 1.810000, 0.100000]

	[4.800000, 0.000000, 1.192500]
	[8.900000, 1.810000, 0.100000]

	[0.991837, 0.500000, 0.767188]
	[0.999864, 0.859362, 0.524979]

	[
	[0.659001, 0.242433, 0.098566]
	]
	[
	[0.576117, 0.211942, 0.211942]
	]
	*/

// 	NOTE: set up thread stuff so we don't have to do it on every layer of
// 	NOTE: the forward pass
// 	TODO: turn this into a function
// 	int NumThreads = 4;
// 	HANDLE* ThreadHandles = (HANDLE*) malloc(NumThreads * sizeof(HANDLE)); 
// 	thread_dense_forward_args* DenseForwardArgs = (thread_dense_forward_args*) (
// 		malloc(NumThreads * sizeof(thread_dense_forward_args))
// 	);
// 	thread_relu_forward_args* ReluForwardArgs = (thread_relu_forward_args*) (
// 		malloc(NumThreads * sizeof(thread_relu_forward_args))
// 	);
// 	thread_sigmoid_forward_args* SigmoidForwardArgs = (
// 		(thread_sigmoid_forward_args*) 
// 		malloc(NumThreads * sizeof(thread_sigmoid_forward_args))
// 	);
// 	thread_softmax_forward_args* SoftmaxForwardArgs = (
// 		(thread_softmax_forward_args*) 
// 		malloc(NumThreads * sizeof(thread_softmax_forward_args))
// 	);
// 	thread_mse_args* MseArgs = (thread_mse_args*) (
// 		malloc(NumThreads * sizeof(thread_mse_args))
// 	);
// 	thread_cross_entropy_args* CrossEntropyArgs = (thread_cross_entropy_args*) (
// 		malloc(NumThreads * sizeof(thread_cross_entropy_args))
// 	);

// 	float Input1Data[4] = {1, 2, 3, 2.5};
// 	float Input2Data[4] = {2.0f, 5.0f, -1.0f, 2.0f};
// 	float Weights1Data[4] = {0.2f, 0.8f, -0.5f, 1.0f};
// 	float Weights2Data[4] = {0.5f, -0.91f, 0.26f, -0.5f};
// 	float Weights3Data[4] = {-0.26f, -0.27f, 0.17f, 0.87f};
// 	float BiasData[3] = {2.0f, 3.0f, 0.5f};

// 	vector_array* Inputs = NULL;
// 	dense_layer* Layer1 = NULL;
// 	dense_layer* Layer2 = NULL;
// 	vector_array* Layer1Outputs = NULL;
// 	vector_array* Layer2Outputs = NULL;	

// 	AllocVectorArray(2, ARRAY_COUNT(Input1Data), &Inputs);
// 	memcpy(GetVector(*Inputs, 0), &Input1Data[0], GetVectorDataSize(*Inputs));
// 	memcpy(GetVector(*Inputs, 1), &Input2Data[0], GetVectorDataSize(*Inputs));

// 	AllocDenseLayer(ARRAY_COUNT(Input1Data), ARRAY_COUNT(BiasData), &Layer1);
// 	weights* Weights = Layer1->Weights;
// 	memcpy(
// 		GetVector(*Weights, 0), &Weights1Data[0], GetVectorDataSize(*Weights)
// 	);
// 	memcpy(
// 		GetVector(*Weights, 1), &Weights2Data[0], GetVectorDataSize(*Weights)
// 	);
// 	memcpy(
// 		GetVector(*Weights, 2), &Weights3Data[0], GetVectorDataSize(*Weights)
// 	);

// 	vector* Biases = Layer1->Biases;
// 	memcpy(Biases->Data, &BiasData[0], GetVectorDataSize(*Biases));

// 	float Layer2Weights1Data[3] = {1, 0, 0};
// 	float Layer2Weights2Data[3] = {0, -1, 0};
// 	float Layer2Weights3Data[3] = {0, 0, 0.5};
// 	AllocDenseLayer(ARRAY_COUNT(BiasData), ARRAY_COUNT(BiasData), &Layer2);
// 	Weights = Layer2->Weights;
// 	memcpy(
// 		GetVector(*Weights, 0),
// 		&Layer2Weights1Data[0],
// 		GetVectorDataSize(*Weights)
// 	);
// 	memcpy(
// 		GetVector(*Weights, 1),
// 		&Layer2Weights2Data[0],
// 		GetVectorDataSize(*Weights)
// 	);
// 	memcpy(
// 		GetVector(*Weights, 2),
// 		&Layer2Weights3Data[0],
// 		GetVectorDataSize(*Weights)
// 	);
// 	Biases = Layer2->Biases;
// 	memset(Biases->Data, 0, GetVectorDataSize(*Biases));

// 	vector_array* Labels = NULL;
// 	AllocVectorArray(2, 3, &Labels);
// 	float* Label = GetVector(*Labels, 0);
// 	memset(Label, 0, GetVectorDataSize(*Labels));
// 	Label[0] = 1.0f;
// 	Label = GetVector(*Labels, 1);
// 	memset(Label, 0, GetVectorDataSize(*Labels));
// 	Label[1] = 1.0f;
	
// 	AllocLayerOutput(Inputs->Length, *Layer1, &Layer1Outputs);
// 	AllocLayerOutput(Inputs->Length, *Layer2, &Layer2Outputs);

// 	DenseForward(
// 		*Inputs,
// 		*Layer1,
// 		Layer1Outputs,
// 		ThreadHandles,
// 		NumThreads,
// 		DenseForwardArgs
// 	);
// 	PrintVectorArray(*Layer1Outputs);
// 	DenseForward(
// 		*Layer1Outputs,
// 		*Layer2,
// 		Layer2Outputs,
// 		ThreadHandles,
// 		NumThreads,
// 		DenseForwardArgs
// 	);
// 	PrintVectorArray(*Layer2Outputs);
// 	ReluForward(
// 		*Layer2Outputs,
// 		Layer2Outputs,
// 		ThreadHandles,
// 		NumThreads,
// 		ReluForwardArgs
// 	);
// 	PrintVectorArray(*Layer2Outputs);
// 	SigmoidForward(
// 		*Layer2Outputs,
// 		Layer2Outputs,
// 		ThreadHandles,
// 		NumThreads,
// 		SigmoidForwardArgs
// 	);
// 	PrintVectorArray(*Layer2Outputs);
// 	float MseResult = MeanSquaredError(
// 		*Layer2Outputs,
// 		*Labels,
// 		ThreadHandles,
// 		NumThreads,
// 		MseArgs
// 	);
// 	printf("%f\n", MseResult);
// 	printf("\n");

// 	printf("RandInit test\n");
// 	dense_layer* RandInitLayer = NULL;
// 	MakeDenseLayer(
// 		ARRAY_COUNT(Input1Data), ARRAY_COUNT(BiasData), &RandInitLayer
// 	);
// 	vector_array* Layer3Outputs = NULL;
// 	AllocLayerOutput(Inputs->Length, *RandInitLayer, &Layer3Outputs);
// 	DenseForward(
// 		*Inputs,
// 		*RandInitLayer,
// 		Layer3Outputs,
// 		ThreadHandles,
// 		NumThreads,
// 		DenseForwardArgs
// 	);
// 	PrintVectorArray(*Layer3Outputs);
// 	printf("\n");

// 	printf("SoftmaxForward test\n");
// 	float SoftmaxData[3] = {2.0f, 1.0f, 0.1f};
// 	float SoftmaxData2[3] = {1.0f, 0.0f, 0.0f};
// 	vector_array* SoftmaxForwardInputs = NULL;
// 	AllocVectorArray(2, 3, &SoftmaxForwardInputs);
// 	memcpy(
// 		GetVector(*SoftmaxForwardInputs, 0),
// 		&SoftmaxData,
// 		GetVectorDataSize(*SoftmaxForwardInputs)
// 	);
// 	memcpy(
// 		GetVector(*SoftmaxForwardInputs, 1),
// 		&SoftmaxData2,
// 		GetVectorDataSize(*SoftmaxForwardInputs)
// 	);
// 	SoftmaxForward(
// 		*SoftmaxForwardInputs,
// 		SoftmaxForwardInputs,
// 		ThreadHandles,
// 		NumThreads,
// 		SoftmaxForwardArgs
// 	);
// 	PrintVectorArray(*SoftmaxForwardInputs);
// 	float CrossEntropyResult = CrossEntropyLoss(
// 		*SoftmaxForwardInputs,
// 		*Labels,
// 		ThreadHandles,
// 		NumThreads,
// 		CrossEntropyArgs
// 	);
// 	printf("%f\n", CrossEntropyResult);
// 	printf("\n");

// 	// NOTE: for this test code, we don't need to free, but it's here if we need
// 	// CONT: it and we might as well test it 
// 	if(Inputs)
// 	{
// 		FreeVectorArray(Inputs);		
// 	}
// 	if(Layer1)
// 	{
// 		FreeDenseLayer(Layer1);
// 	}
// 	if(Layer2)
// 	{
// 		FreeDenseLayer(Layer2);
// 	}	
// 	if(Layer1Outputs)
// 	{
// 		FreeVectorArray(Layer1Outputs);
// 	}
// 	if(Layer2Outputs)
// 	{
// 		FreeVectorArray(Layer2Outputs);	
// 	}
// 	if(Layer3Outputs)
// 	{
// 		FreeVectorArray(Layer3Outputs);
// 	}

// #if 0
// 	printf("\n");
// 	vector_array* SpiralInputs = NULL;
// 	vector_array* SpiralOutputs = NULL;
// 	int PointsPerClass = 100;
// 	int NumClasses = 3;
// 	int Dimensions = 2;
// 	MakeSpiralData(
// 		PointsPerClass, Dimensions, NumClasses, &SpiralInputs, &SpiralOutputs
// 	);
// 	for(int ClassIndex = 0; ClassIndex < NumClasses; ClassIndex++)
// 	{
// 		for(int DataIndex = 0; DataIndex < SpiralInputs->Length; DataIndex++)
// 		{
// 			float* SpiralOutput = GetVector(*SpiralOutputs, DataIndex);
// 			int Classification = ArgMax(
// 				SpiralOutput, SpiralOutputs->VectorLength
// 			);
// 			if(Classification == ClassIndex)
// 			{
// 				float* SpiralInput = GetVector(*SpiralInputs, DataIndex); 
// 				for(
// 					int ElementIndex = 0;
// 					ElementIndex < Dimensions;
// 					ElementIndex++
// 				)
// 				{
// 					printf("%f,", SpiralInput[ElementIndex]);
// 				}
// 				printf("%d", Classification);
// 				printf("\n");
// 			}
// 		}
// 		printf("\n\n");
// 	}
// #endif

error: 
	return 0;
}