#ifndef NEURAL_NET_COMMON_H
#include <stdint.h>
#include <math.h>

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

int ArgMax(float* Array, uint64_t ArrayLength)
{
	ASSERT(Array != NULL);
	ASSERT(ArrayLength > 0);
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

#define PI32 3.14159
float RandFloat()
{
	return ((float) rand()) / ((float) RAND_MAX);
}

float RandGaussian()
{
	// NOTE: returns a float based on a gaussian distribution centered at 0 
	// CONT: and a standard deviation of 1
	// CONT: uses the Box-Muller algorithm
	
	float U1 = RandFloat();
	float U2 = RandFloat();
	return (float) (sqrt(-2.0f * log(U1)) * cos(2 * PI32 * U2)); 
}

void InitSpiralData(
	int PointsPerClass,
	int Dimensions,
	int NumClasses,
	vector_array* Inputs,
	vector_array* Outputs
)
{
	float RadiusIncrement = 1.0f / PointsPerClass;
	for(int Class = 0; Class < NumClasses; Class++)
	{
		float Radius = 0.0f;
		float Theta = 4.0f * Class;
		float ThetaIncrement = 4.0f / ((float) PointsPerClass);
		for(int Point = 0; Point < PointsPerClass; Point++)
		{
			float* Output = GetVector(*Outputs, Class * PointsPerClass + Point);
			Output[Class] = 1.0f;

			float* Input = GetVector(*Inputs, Class * PointsPerClass + Point);
			for(int ElementIndex = 0; ElementIndex < Dimensions; ElementIndex++)
			{
				float ElementTheta = Theta + (0.2f * RandGaussian());
				if(ElementIndex % 2 == 0)
				{
					Input[ElementIndex] = Radius * ((float) sin(ElementTheta));
				}
				else
				{
					Input[ElementIndex] = Radius * ((float) cos(ElementTheta));
				}
			}
			Theta += ThetaIncrement;
			Radius += RadiusIncrement;
		}
	}
}

#define NEURAL_NET_COMMON_H
#endif 