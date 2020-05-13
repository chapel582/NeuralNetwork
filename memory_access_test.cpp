/*
NOTE: 
this file is a speed test to compare allocation and access times for 
two different paradigms

Paradigm one: 
	malloc call for each pointer 

Paradigm two:
	
*/
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <windows.h>

// SECTION START: Performance inspection 
int64_t GlobalPerformanceFrequency = 0;
inline int64_t GetWallClock(void)
{
	LARGE_INTEGER Result;
	// NOTE: QueryPerformanceCounter gets wall clock time
	QueryPerformanceCounter(&Result);
	return Result.QuadPart;
}

inline float GetSecondsElapsed(int64_t Start, int64_t End)
{
	float Result;
	Result = (
		((float) (End - Start)) / 
		((float) GlobalPerformanceFrequency)
	);
	return Result;
}
// SECTION STOP: Performance inspection 

// SECTION START: Flat array of arrays
inline uint8_t* AllocArrayOfArrays(uint64_t NumArrays, uint64_t NumElements)
{
	return (uint8_t*) malloc(NumArrays * NumElements * sizeof(uint8_t));
}

inline uint8_t* GetArray(
	uint8_t* ArrayOfArrays, uint64_t ArrayIndex, uint64_t NumElements)
{
	return ArrayOfArrays + ArrayIndex * NumElements;
}

inline uint8_t* GetElement(
	uint8_t* ArrayOfArrays,
	uint64_t ArrayIndex,
	uint64_t NumElements,
	uint64_t ElementIndex
)
{
	return ArrayOfArrays + ArrayIndex * NumElements + ElementIndex;
}
// SECTION STOP: Flat array of arrays

int main(void)
{
	LARGE_INTEGER PerformanceFrequency;
	QueryPerformanceFrequency(&PerformanceFrequency);
	GlobalPerformanceFrequency = PerformanceFrequency.QuadPart;

	uint64_t NumElements = 100;
	uint64_t NumArrays = 1000000;

	int64_t Start = GetWallClock();
	uint8_t** ArrayOfArrays = (uint8_t**) malloc(NumArrays * sizeof(uint8_t*));
	for(int Index = 0; Index < NumArrays; Index++)
	{
		ArrayOfArrays[Index] = (uint8_t*) malloc(NumElements * sizeof(uint8_t));
	}
	int64_t End = GetWallClock();
	float SecondsElapsed = GetSecondsElapsed(Start, End);
	printf("Time to malloc for every pointer: %f\n", SecondsElapsed);

	Start = GetWallClock();
	uint8_t* FlatArrayOfArrays = AllocArrayOfArrays(NumArrays, NumElements);
	End = GetWallClock();
	SecondsElapsed = GetSecondsElapsed(Start, End);
	printf("Time to malloc flat array of arrays: %f\n", SecondsElapsed);

	uint64_t Sum = 0;
	Start = GetWallClock();
	for(int ArrayIndex = 0; ArrayIndex < NumArrays; ArrayIndex++)
	{

		for(int ElementIndex = 0; ElementIndex < NumElements; ElementIndex++)
		{
			Sum += ArrayOfArrays[ArrayIndex][ElementIndex];			
		}
	}
	End = GetWallClock();
	SecondsElapsed = GetSecondsElapsed(Start, End);
	printf("Time to sum all elements of all arrays: %f\n", SecondsElapsed);

	Sum = 0;
	Start = GetWallClock();
	for(int ArrayIndex = 0; ArrayIndex < NumArrays; ArrayIndex++)
	{
		for(int ElementIndex = 0; ElementIndex < NumElements; ElementIndex++)
		{
			Sum += *GetElement(
				FlatArrayOfArrays,
				ArrayIndex,
				NumElements,
				ElementIndex
			);
		}
	}
	End = GetWallClock();
	SecondsElapsed = GetSecondsElapsed(Start, End);
	printf(
		"Time to sum all elements of all arrays (flat): %f\n", SecondsElapsed
	);

	Sum = 0;
	Start = GetWallClock();
	for(int ArrayIndex = 0; ArrayIndex < NumArrays; ArrayIndex++)
	{
		uint8_t* Array = ArrayOfArrays[ArrayIndex];
		for(int ElementIndex = 0; ElementIndex < NumElements; ElementIndex++)
		{
			Sum += Array[ElementIndex];			
		}
	}
	End = GetWallClock();
	SecondsElapsed = GetSecondsElapsed(Start, End);
	printf(
		"Time to sum all elements of all arrays, single access: %f\n",
		SecondsElapsed
	);

	Sum = 0;
	Start = GetWallClock();
	for(int ArrayIndex = 0; ArrayIndex < NumArrays; ArrayIndex++)
	{
		uint8_t* Array = GetArray(FlatArrayOfArrays, ArrayIndex, NumElements);
		for(int ElementIndex = 0; ElementIndex < NumElements; ElementIndex++)
		{
			Sum += Array[ElementIndex];
		}
	}
	End = GetWallClock();
	SecondsElapsed = GetSecondsElapsed(Start, End);
	printf(
		"Time to sum all elements of all arrays (flat), single access: %f\n",
		SecondsElapsed
	);

	Sum = 0;
	uint8_t* LastElement = GetElement(
		FlatArrayOfArrays,
		NumArrays - 1,
		NumElements,
		NumElements - 1
	);
	Start = GetWallClock();
	for(
		uint8_t* Element = GetArray(FlatArrayOfArrays, 0, NumElements);
		Element <= LastElement;
		Element++
	)
	{
		Sum += *Element;
	}
	End = GetWallClock();
	SecondsElapsed = GetSecondsElapsed(Start, End);
	printf(
		"Time to sum all elements of all arrays (flat), pointer movement: %f\n",
		SecondsElapsed
	);
}