#include "matrix.h"

#include <stdio.h>

inline bool MatricesAreEquivalent(matrix* M1, matrix* M2)
{
	if(!(M1->NumRows == M2->NumRows && M1->NumColumns == M2->NumColumns))
	{
		return false;
	}
	return memcmp(M1->Data, M2->Data, GetMatrixDataSize(M1)) == 0;
}

inline void MatrixClear(matrix* Matrix)
{
	memset(
		Matrix->Data, 0, Matrix->NumRows * Matrix->NumColumns * sizeof(float)
	);
}

void WriteMatrix(matrix* Matrix, FILE* File)
{
	fwrite(Matrix->Data, 1, GetMatrixDataSize(Matrix), File);
}

void SaveMatrix(matrix* Matrix, char* FilePath)
{
	// NOTE: a way to save matrix data to a file for use in unit tests
	FILE* File;
	fopen_s(&File, FilePath, "wb");
	WriteMatrix(Matrix, File);
	fclose(File);
}

bool LoadMatrix(matrix* Matrix, char* FilePath)
{
	// NOTE: a way to load a matrix from a file for comparison to unit test 
	// CONT: results
	FILE* File;
	fopen_s(&File, FilePath, "rb");
	size_t BytesRead = fread(Matrix->Data, 1, GetMatrixDataSize(Matrix), File);
	fclose(File);
	return BytesRead == GetMatrixDataSize(Matrix);
}

void FillIdentityMatrix(matrix* Matrix)
{
	for(uint32_t Row = 0; Row < Matrix->NumRows; Row++)
	{
		for(uint32_t Col = 0; Col < Matrix->NumColumns; Col++)
		{
			float Value;
			if(Row == Col)
			{
				Value = 1.0f;
			}
			else
			{
				Value = 0.0f;
			}
			SetMatrixElement(Matrix, Row, Col, Value);
		}
	}
}

HOST_PREFIX DEVICE_PREFIX
uint32_t GetMatrixArrayCount(matrix* Matrix)
{
	return Matrix->NumRows * Matrix->NumColumns;
}

HOST_PREFIX DEVICE_PREFIX
size_t GetMatrixDataSize(matrix* Matrix)
{
	return GetMatrixArrayCount(Matrix) * sizeof(float);
}

HOST_PREFIX DEVICE_PREFIX
float* GetMatrixRow(matrix* Matrix, uint32_t Row)
{
	assert(Row < Matrix->NumRows);
	float* Element = Matrix->Data + Row * Matrix->NumColumns;
	return Element;
}

HOST_PREFIX DEVICE_PREFIX
float GetMatrixElement(matrix* Matrix, uint32_t Row, uint32_t Column)
{
	assert(Row < Matrix->NumRows);
	assert(Column < Matrix->NumColumns);
	float* Element = Matrix->Data + Row * Matrix->NumColumns + Column;
	return *Element;
}

HOST_PREFIX DEVICE_PREFIX
float GetMatrixElement(matrix* Matrix, uint32_t ElementIndex)
{
	// NOTE: made available if the Row, Column asserts in the standard 
	// CONT: GetMatrixElement isn't needed. Mostly used for when you don't care
	// CONT: if you have a row or column matrix
	assert(ElementIndex < GetMatrixArrayCount(Matrix));
	float* Element = Matrix->Data + ElementIndex;
	return *Element;
}

HOST_PREFIX DEVICE_PREFIX
void SetMatrixElement(
	matrix* Matrix, uint32_t Row, uint32_t Column, float Value
)
{
	assert(Row < Matrix->NumRows);
	assert(Column < Matrix->NumColumns);
	float* Element = Matrix->Data + Row * Matrix->NumColumns + Column;
	*Element = Value;
}

HOST_PREFIX DEVICE_PREFIX
void SetMatrixElement(
	matrix* Matrix, uint32_t ElementIndex, float Value
)
{
	// NOTE: made available if the Row, Column asserts in the standard 
	// CONT: GetMatrixElement isn't needed. Mostly used for when you don't care
	// CONT: if you have a row or column matrix
	assert(ElementIndex < GetMatrixArrayCount(Matrix));
	float* Element = Matrix->Data + ElementIndex;
	*Element = Value;
}