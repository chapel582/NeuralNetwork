#ifndef NEURAL_NET_COMMON_H

#define ARRAY_COUNT(Array) (sizeof(Array) / sizeof(Array[0]))
#define ASSERT(Expression) if(!(Expression)) {*(int*) 0 = 0;}

struct vector
{
	uint64_t Length;
	float* Data;	
};

inline size_t GetVectorDataSize(vector Vector)
{
	return Vector.Length * sizeof(float);
}

struct vector_array
{
	uint64_t Length;
	uint64_t VectorLength;
	float* Vectors;
};

inline size_t GetVectorArrayDataSize(vector_array VectorArray)
{
	return VectorArray.VectorLength * VectorArray.Length * sizeof(float);
}

inline size_t GetVectorDataSize(vector_array VectorArray)
{
	return VectorArray.VectorLength * sizeof(float);
}

inline float* GetFirstVector(vector_array VectorArray)
{
	return VectorArray.Vectors;
}

inline float* GetVector(vector_array VectorArray, uint64_t Index)
{
	ASSERT(Index < VectorArray.Length);
	return VectorArray.Vectors + Index * VectorArray.VectorLength;
}

void PrintVectorArray(vector_array VectorArray)
{
	float* Element = &VectorArray.Vectors[0];
	printf("[\n");
	for(int VectorIndex = 0; VectorIndex < VectorArray.Length; VectorIndex++)
	{
		printf("[");
		for(
			int ElementIndex = 0;
			ElementIndex < (VectorArray.VectorLength - 1);
			ElementIndex++
		)
		{
			printf("%f, ", *Element++);
		}
		printf("%f]\n", *Element++);
	}
	printf("]\n");
}

typedef vector_array weights;

struct dense_layer
{
	weights* Weights;
	vector* Biases;
};

void FloatCopy(float* Destination, float* Source, uint64_t Length)
{
	float* OnePastFinalFloat = Destination + Length;
	while(Destination < OnePastFinalFloat)
	{
		*Destination = *Source;
		Destination++;
		Source++;  
	}
}

void FloatSet(float* Destination, float Value, uint64_t Length)
{
	float* OnePastFinalFloat = Destination + Length;
	while(Destination < OnePastFinalFloat)
	{
		*Destination = Value;
		Destination++;
	}
}

#define NEURAL_NET_COMMON_H
#endif 