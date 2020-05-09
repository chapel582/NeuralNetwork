#ifndef NEURAL_NET_COMMON_H

#define ARRAY_COUNT(Array) (sizeof(Array) / sizeof(Array[0]))
#define ASSERT(Expression) if(!(Expression)) {*(int*) 0 = 0;}

struct vector
{
	uint64_t Length;
	float* Data;	
};

struct vector_array
{
	uint64_t Length;
	vector* Vectors;
};

void PrintVectorArray(vector_array VectorArray)
{
	printf("[\n");
	for(int VectorIndex = 0; VectorIndex < VectorArray.Length; VectorIndex++)
	{
		vector Vector = VectorArray.Vectors[VectorIndex];
		printf("[");
		int ElementIndex;
		for(
			ElementIndex = 0;
			ElementIndex < (Vector.Length - 1);
			ElementIndex++
		)
		{
			printf("%f, ", Vector.Data[ElementIndex]);
		}
		printf("%f]\n", Vector.Data[ElementIndex]);
	}
	printf("]\n");
}

typedef vector_array weights;

#define NEURAL_NET_COMMON_H
#endif 