#ifndef ARG_MAX_H

#include <stdint.h>

int ArgMax(float* Array, uint64_t ArrayLength)
{
	int Index = 0;
	int Result = Index;
	float Highest = Array[Index];
	for(Index = 1; Index < ArrayLength; Index++)
	{
		if(Array[Index] > Highest)
		{
			Highest = Array[Index];
			Result = Index;
		}
	}
	return Result;
}

#define ARG_MAX_H

#endif