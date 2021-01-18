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

#pragma pack(push, 1)
struct tensor_header
{
	uint32_t DimCount;
	// TODO: add endianness check here
};
#pragma pack(pop)

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

float* GetElementVarArgs(float_tensor* Tensor, va_list VarArgs)
{
	float* Data = Tensor->Data;
	for(uint32_t CurrentDim = 0; CurrentDim < Tensor->DimCount; CurrentDim++)
	{
		uint32_t Index = va_arg(VarArgs, uint32_t);
		Data += Index * Tensor->Strides[CurrentDim];
	}

	va_end(VarArgs);

	return Data;
}

float* GetElementPtr(float_tensor* Tensor, ...)
{
	va_list VarArgs;
	va_start(VarArgs, Tensor);
	return GetElementVarArgs(Tensor, VarArgs);
}

float GetElement(float_tensor* Tensor, ...)
{
	va_list VarArgs;
	va_start(VarArgs, Tensor);
	return *GetElementVarArgs(Tensor, VarArgs);
}

void SetElement(float_tensor* Tensor, float Value, ...)
{
	va_list VarArgs;
	va_start(VarArgs, Value);
	float* ElementPtr = GetElementVarArgs(Tensor, VarArgs);
	*ElementPtr = Value;
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
	else
	{
		if(Tensor1->DimCount == 0)
		{
			return true;
		}
		else
		{
			int Cmp = memcmp(
				Tensor1->Shape,
				Tensor2->Shape,
				Tensor1->DimCount * sizeof(uint32_t)
			);
			return Cmp == 0;
		}
	}
}

uint32_t GetTensorElementOffset(float_tensor* Tensor, uint32_t ElementIndex)
{
	// NOTE: for accessing the Nth element of a tensor
	uint32_t Offset = 0;
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
		Offset += DimIndex * Tensor->Strides[CurrentDim];
		ElementsInDimension *= Tensor->Shape[CurrentDim];
	}

	return Offset;
}

void ScalarMult(float_tensor* Result, float_tensor* Input, float Scalar)
{
	assert(IsSameShape(Result, Input));
	uint32_t TotalElements = GetTotalElements(Input);
	for(
		uint32_t ElementIndex = 0;
		ElementIndex < TotalElements;
		ElementIndex++
	)
	{
		uint32_t Offset = GetTensorElementOffset(Input, ElementIndex);
		Result->Data[Offset] = Scalar * Input->Data[Offset];
	}
}

void ScalarAdd(float_tensor* Result, float_tensor* Input, float Scalar)
{
	assert(IsSameShape(Result, Input));
	uint32_t TotalElements = GetTotalElements(Input);
	for(
		uint32_t ElementIndex = 0;
		ElementIndex < TotalElements;
		ElementIndex++
	)
	{
		uint32_t Offset = GetTensorElementOffset(Input, ElementIndex);
		Result->Data[Offset] = Input->Data[Offset] + Scalar;
	}
}

typedef float float_to_float_function(float Arg);

void SetTensorElements(float_tensor* Input, float Value)
{
	uint32_t TotalElements = GetTotalElements(Input);
	for(
		uint32_t ElementIndex = 0;
		ElementIndex < TotalElements;
		ElementIndex++
	)
	{
		uint32_t Offset = GetTensorElementOffset(Input, ElementIndex);
		Input->Data[Offset] = Value;
	}
}

void CopyTensorElements(float_tensor* Destination, float_tensor* Source)
{
	assert(IsSameShape(Destination, Source));
	uint32_t TotalElements = GetTotalElements(Source);
	for(
		uint32_t ElementIndex = 0;
		ElementIndex < TotalElements;
		ElementIndex++
	)
	{
		uint32_t SourceOffset = GetTensorElementOffset(Source, ElementIndex);
		uint32_t DestinationOffset = GetTensorElementOffset(
			Destination, ElementIndex
		);
		Destination->Data[DestinationOffset] = Source->Data[SourceOffset];
	}
}

void OneTensorBroadcast(
	float_tensor* Result,
	float_tensor* Input,
	float_to_float_function* FtfFunction
)
{
	assert(IsSameShape(Result, Input));
	uint32_t TotalElements = GetTotalElements(Input);
	for(
		uint32_t ElementIndex = 0;
		ElementIndex < TotalElements;
		ElementIndex++
	)
	{
		uint32_t Offset = GetTensorElementOffset(Input, ElementIndex);
		Result->Data[Offset] = FtfFunction(Input->Data[Offset]);
	}
}

bool AreTensorsEquivalent(float_tensor* T1, float_tensor* T2)
{
	assert(IsSameShape(T1, T2));

	uint32_t TotalElements = GetTotalElements(T1);
	for(
		uint32_t ElementIndex = 0;
		ElementIndex < TotalElements;
		ElementIndex++
	)
	{
		uint32_t T1Offset = GetTensorElementOffset(T1, ElementIndex);
		uint32_t T2Offset = GetTensorElementOffset(T2, ElementIndex);
		if(T1->Data[T1Offset] != T2->Data[T2Offset])
		{
			return false;
		}
	}

	return true;
}

typedef float two_arg_ftf(float Arg1, float Arg2);
void TwoTensorBroadcast(
	float_tensor* Result,
	float_tensor* T1,
	float_tensor* T2,
	two_arg_ftf FtfFunction
)
{
	assert(IsSameShape(Result, T1));
	assert(IsSameShape(Result, T2));

	uint32_t TotalElements = GetTotalElements(T1);
	for(
		uint32_t ElementIndex = 0;
		ElementIndex < TotalElements;
		ElementIndex++
	)
	{
		uint32_t ResultOffset = GetTensorElementOffset(Result, ElementIndex);
		uint32_t T1Offset = GetTensorElementOffset(T1, ElementIndex);
		uint32_t T2Offset = GetTensorElementOffset(T2, ElementIndex);
		Result->Data[ResultOffset] = FtfFunction(
			T1->Data[T1Offset], T2->Data[T2Offset]
		);
	}
}

inline uint32_t GetTensorDataSize(float_tensor* Tensor)
{
	// NOTE: only works with contiguous memory
	uint32_t Result = 1;
	uint32_t* Shape = Tensor->Shape; 
	for(uint32_t Index = 0; Index < Tensor->DimCount; Index++)
	{
		Result *= Shape[Index];
	}

	return Result * sizeof(*Tensor->Data);
}

// TODO: TransposeCopy

#define TENSOR_H

#endif