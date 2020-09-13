#include "neural_net.h"
#include "matrix.h"

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