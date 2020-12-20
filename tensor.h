#ifndef TENSOR_H

#include "project_flags.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <assert.h>

struct float_tensor
{
	uint32_t DimCount;

	uint32_t* Shape;
	uint32_t* Strides;
	float* Data;
};

inline uint32_t GetTotalElements(float_tensor* Tensor)
{
	uint32_t TotalElements = 1;
	for(uint32_t Index = 0; Index < Tensor->DimCount; Index++)
	{
		TotalElements *= Tensor->Shape[Index];
	}

	return TotalElements;
}

void PrintTensor(float_tensor* Tensor)
{
	// NOTE: our print tensor just prints the last dimension as a 1d array
	uint32_t TotalElements = GetTotalElements(Tensor);
	uint32_t Stride = Tensor->Strides[Tensor->DimCount - 1];
	for(
		uint32_t ElementIndex = 0;
		ElementIndex < TotalElements;
		ElementIndex++
	)
	{
		uint32_t DataIndex = ElementIndex * Stride;
		printf("%f, ", Tensor->Data[DataIndex]);
		if(ElementIndex % Tensor->Shape[Tensor->DimCount - 1] == 0)
		{
			printf("\n");
		}
	}
}
// tensor<> TransposeView();
// TODO: TransposeCopy

#define TENSOR_H

#endif