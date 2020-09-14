#include "int_shuffler.h"

#include <stdlib.h>

int_shuffler MakeIntShuffler(uint32_t Range)
{
	int_shuffler IntShuffler = {};
	IntShuffler.Range = Range;
	IntShuffler.Cells = (linked_int*) malloc(sizeof(linked_int) * Range);
	IntShuffler.Result = (int*) malloc(sizeof(int) * Range);
	return IntShuffler;
}

void FreeIntShuffler(int_shuffler IntShuffler)
{
	free(IntShuffler.Cells);
	free(IntShuffler.Result);
}

void ShuffleInts(int_shuffler* IntShuffler)
{
	// NOTE: needed for mini batch shuffling
	linked_int_list* List = &IntShuffler->List;
	List->Length = 0;

	linked_int* Previous = IntShuffler->Cells + 0;
	Previous->Value = 0;
	List->Head = Previous;
	List->Length++;
	linked_int* Current = Previous;
	for(uint32_t Index = 1; Index < IntShuffler->Range; Index++)
	{
		Current = IntShuffler->Cells + Index;
		Current->Value = Index;

		Previous->Next = Current;
		Previous = Current;
		List->Length++;
	}
	Current->Next = NULL;

	int ArrayIndex = 0;
	for(uint32_t Index = 0; Index < IntShuffler->Range; Index++)
	{
		int Value = rand() % List->Length;
		Current = List->Head;
		for(int LinkIndex = 0; LinkIndex < Value; LinkIndex++)	
		{
			Previous = Current;
			Current = Current->Next;
		}
		if(Current == List->Head)
		{
			List->Head = Current->Next;
		}
		else
		{
			Previous->Next = Current->Next;
		}

		List->Length--;
		IntShuffler->Result[ArrayIndex++] = Current->Value;
	}
}
