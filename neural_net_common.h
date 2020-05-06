#ifndef NEURAL_NET_COMMON_H

#define ARRAY_COUNT(Array) (sizeof(Array) / sizeof(Array[0]))
#define ASSERT(Expression) if(!(Expression)) {*(int*) 0 = 0;}

struct vector
{
	uint64_t Length;
	float* Data;	
};

struct weights
{
	uint64_t Length;
	vector* Vectors;
};

#define NEURAL_NET_COMMON_H
#endif 