#ifndef INT_SHUFFLER_H

#include <stdint.h>

struct linked_int;
struct linked_int
{
	int Value;
	linked_int* Next;
};

struct linked_int_list
{
	linked_int* Head;
	uint32_t Length;
};

struct int_shuffler
{
	linked_int_list List;
	uint32_t Range;
	linked_int* Cells;
	int* Result;
};

int_shuffler MakeIntShuffler(uint32_t Range);
void FreeIntShuffler(int_shuffler IntShuffler);
void ShuffleInts(int_shuffler* IntShuffler);

#define INT_SHUFFLER_H
#endif