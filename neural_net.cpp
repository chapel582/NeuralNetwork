#include "neural_net.h"
#include "matrix.h"
#include "int_shuffler.h"

#include <stdint.h>

inline float RandUnity()
{
	// NOTE: returns random float between 0.0 and 1.0 
	return ((float) (rand() % RAND_MAX)) / ((float) RAND_MAX);
}

void FillRandomMatrix(matrix* Matrix, float Range)
{
	// NOTE: fills matrix with values randomly between 0.0f and Range
	// NOTE: values should never be 0.0f exactly
	for(uint32_t Row = 0; Row < Matrix->NumRows; Row++)
	{
		for(uint32_t Col = 0; Col < Matrix->NumColumns; Col++)
		{
			float Value;
			do
			{
				// NOTE: need small values to prevent runaway
				Value = Range * RandUnity();
			} while(Value == 0.0f);
			SetMatrixElement(Matrix, Row, Col, Value);
		}
	}
}

void InitDenseLayers(neural_net* NeuralNet)
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
				FillRandomMatrix(
					&DenseLayer->Weights,
					(
						1.0f / (
							DenseLayer->Weights.NumRows + 
							DenseLayer->Weights.NumColumns
						)
					)
				);
				FillRandomMatrix(&DenseLayer->Bias, 0.001f);
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

HOST_PREFIX DEVICE_PREFIX
void ReluForwardCore(
	matrix* M1, matrix* Result, uint32_t Start, uint32_t Stride
)
{
	uint32_t NumResultElements = GetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		float NewValue;
		float OldValue = GetMatrixElement(M1, ResultIndex);
		if(OldValue < 0)
		{
			NewValue = 0;
		}
		else
		{
			NewValue = OldValue;
		}
		SetMatrixElement(Result, ResultIndex, NewValue);
	}
}

HOST_PREFIX DEVICE_PREFIX
void ReluBackCore(
	matrix* Inputs,
	matrix* NextLayerGradient,
	matrix* LayerGradient,
	uint32_t Start,
	uint32_t Stride
)
{
	assert(
		GetMatrixArrayCount(LayerGradient) == 
		GetMatrixArrayCount(NextLayerGradient)
	);
	uint32_t NumResultElements = GetMatrixArrayCount(LayerGradient);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		float LayerGradientElement;
		float InputValue = GetMatrixElement(Inputs, ResultIndex);
		if(InputValue <= 0)
		{
			LayerGradientElement = 0;
		}
		else
		{
			LayerGradientElement = GetMatrixElement(
				NextLayerGradient, ResultIndex
			);
		}
		SetMatrixElement(LayerGradient, ResultIndex, LayerGradientElement);
	}
}

HOST_PREFIX DEVICE_PREFIX
float MseForwardCore(
	matrix* Predictions, matrix* Labels, uint32_t Start, uint32_t Stride
)
{
	float Result = 0.0f;
	uint32_t NumRows = Predictions->NumRows;
	uint32_t NumColumns = Predictions->NumColumns;
	for(uint32_t Row = Start; Row < NumRows; Row += Stride)
	{
		for(uint32_t Col = 0; Col < NumColumns; Col++)
		{
			float Difference = (
				GetMatrixElement(Predictions, Row, Col) - 
				GetMatrixElement(Labels, Row, Col)
			);
			Result += Difference * Difference;
		}
	}

	return Result;
}

HOST_PREFIX DEVICE_PREFIX
void CreateMiniBatch(
	int_shuffler* IntShuffler,
	matrix* MiniBatchData,
	matrix* MiniBatchLabels,
	matrix* Inputs,
	matrix* Labels,
	uint32_t BatchIndex,
	uint32_t MiniBatchSize,
	uint32_t Start,
	uint32_t Stride
)
{
	// NOTE: create mini batch
	uint32_t IndexHandleStart = BatchIndex * MiniBatchSize;
	for(
		uint32_t IndexHandle = IndexHandleStart + Start;
		IndexHandle < (IndexHandleStart + MiniBatchSize);
		IndexHandle += Stride
	)
	{
		int RowToGet = IntShuffler->Result[IndexHandle];
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
}