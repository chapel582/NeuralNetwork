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

float GetElement(
	float_tensor* Tensor, uint32_t* Indices, uint32_t IndexCount
)
{
	assert(IndexCount <= Tensor->DimCount);
	float* Data = Tensor->Data;
	for(uint32_t CurrentDim = 0; CurrentDim < IndexCount; CurrentDim++)
	{
		assert(Indices[CurrentDim] < Tensor->Shape[CurrentDim]);
		Data += Indices[CurrentDim] * Tensor->Strides[CurrentDim];
	}

	return *Data;
}

float GetElement(float_tensor* Tensor, ...)
{
	va_list VarArgs;
	va_start(VarArgs, Tensor);

	float* Data = Tensor->Data;
	for(uint32_t CurrentDim = 0; CurrentDim < Tensor->DimCount; CurrentDim++)
	{
		uint32_t Index = va_arg(VarArgs, uint32_t);
		Data += Index * Tensor->Strides[CurrentDim];
	}

	va_end(VarArgs);

	return *Data;
}

void PrintTensor(float_tensor* Tensor)
{
	// NOTE: our print tensor just prints the last dimension as a 1d array
	
	if(Tensor->DimCount > 0)
	{
		uint32_t TotalElements = GetTotalElements(Tensor);
		for(
			uint32_t ElementIndex = 0;
			ElementIndex < TotalElements;
			ElementIndex++
		)
		{
			float* Data = Tensor->Data;
			uint32_t ElementsInDimension = 1;
			for(
				int32_t CurrentDim = Tensor->DimCount - 1;
				CurrentDim >= 0;
				CurrentDim--
			)
			{
				uint32_t DimIndex = (
					(ElementIndex / ElementsInDimension) %
					Tensor->Shape[CurrentDim]
				);
				Data += DimIndex * Tensor->Strides[CurrentDim];
				ElementsInDimension *= Tensor->Shape[CurrentDim];
			}
			printf("%f, ", *Data);
			if(
				ElementIndex % Tensor->Shape[Tensor->DimCount - 1] == 
				(Tensor->Shape[Tensor->DimCount - 1] - 1)
			)
			{
				printf("\n");
			}
		}
	}
	else
	{
		printf("%f", Tensor->Data[0]);
	}

	printf("\n");
}

inline bool IsSameShape(float_tensor* Tensor1, float_tensor* Tensor2)
{
	if(Tensor1->DimCount != Tensor2->DimCount)
	{
		return false;
	}
	int Cmp = memcmp(
		Tensor1->Shape, Tensor2->Shape, Tensor1->DimCount * sizeof(uint32_t)
	);
	return Cmp == 0;
}

void ScalarMult(float_tensor* Result, float_tensor* Input, float Scalar)
{
	assert(IsSameShape(Result, Input));
	if(Input->DimCount > 0)
	{
		uint32_t TotalElements = GetTotalElements(Input);
		for(
			uint32_t ElementIndex = 0;
			ElementIndex < TotalElements;
			ElementIndex++
		)
		{
			uint32_t Offset = 0;
			uint32_t ElementsInDimension = 1;
			for(
				int32_t CurrentDim = Input->DimCount - 1;
				CurrentDim >= 0;
				CurrentDim--
			)
			{
				uint32_t DimIndex = (
					(ElementIndex / ElementsInDimension) %
					Input->Shape[CurrentDim]
				);
				Offset += DimIndex * Input->Strides[CurrentDim];
				ElementsInDimension *= Input->Shape[CurrentDim];
			}
			Result->Data[Offset] = Scalar * Input->Data[Offset];
		}
	}
	else
	{
		Result->Data[0] = Scalar * Input->Data[0];
	}
}

typedef float float_to_float_function(float Arg);

void OneTensorBroadcast(
	float_tensor* Result,
	float_tensor* Input,
	float_to_float_function* FtfFunction
)
{
	assert(IsSameShape(Result, Input));
	if(Input->DimCount > 0)
	{
		uint32_t TotalElements = GetTotalElements(Input);
		for(
			uint32_t ElementIndex = 0;
			ElementIndex < TotalElements;
			ElementIndex++
		)
		{
			uint32_t Offset = 0;
			uint32_t ElementsInDimension = 1;
			for(
				int32_t CurrentDim = Input->DimCount - 1;
				CurrentDim >= 0;
				CurrentDim--
			)
			{
				uint32_t DimIndex = (
					(ElementIndex / ElementsInDimension) %
					Input->Shape[CurrentDim]
				);
				Offset += DimIndex * Input->Strides[CurrentDim];
				ElementsInDimension *= Input->Shape[CurrentDim];
			}
			Result->Data[Offset] = FtfFunction(Input->Data[Offset]);
		}
	}
	else
	{
		Result->Data[0] = FtfFunction(Input->Data[0]);
	}
}

// TODO: TransposeCopy

#define TENSOR_H

#endif