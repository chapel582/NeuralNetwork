#ifndef MATRIX_H

#include <string.h>
#include <stdint.h>
#include <assert.h>

struct matrix
{
	uint32_t NumRows;
	uint32_t NumColumns;
	float* Data;
};

inline size_t GetMatrixDataSize(matrix* Matrix)
{
	return Matrix->NumRows * Matrix->NumColumns * sizeof(float);
}

inline float* GetMatrixRow(matrix* Matrix, uint32_t Row)
{
	assert(Row < Matrix->NumRows);
	float* Element = Matrix->Data + Row * Matrix->NumColumns;
	return Element;
}

inline float GetMatrixElement(matrix* Matrix, uint32_t Row, uint32_t Column)
{
	assert(Row < Matrix->NumRows);
	assert(Column < Matrix->NumColumns);
	float* Element = Matrix->Data + Row * Matrix->NumColumns + Column;
	return *Element;
}

inline float GetMatrixElement(matrix* Matrix, uint32_t ElementIndex)
{
	// NOTE: made available if the Row, Column asserts in the standard 
	// CONT: GetMatrixElement isn't needed. Mostly used for when you don't care
	// CONT: if you have a row or column matrix
	assert(ElementIndex < (Matrix->NumRows * Matrix->NumColumns));
	float* Element = Matrix->Data + ElementIndex;
	return *Element;
}

inline void SetMatrixElement(
	matrix* Matrix, uint32_t Row, uint32_t Column, float Value
)
{
	assert(Row < Matrix->NumRows);
	assert(Column < Matrix->NumColumns);
	float* Element = Matrix->Data + Row * Matrix->NumColumns + Column;
	*Element = Value;
}

inline void SetMatrixElement(
	matrix* Matrix, uint32_t ElementIndex, float Value
)
{
	// NOTE: made available if the Row, Column asserts in the standard 
	// CONT: GetMatrixElement isn't needed. Mostly used for when you don't care
	// CONT: if you have a row or column matrix
	assert(ElementIndex < (Matrix->NumRows * Matrix->NumColumns));
	float* Element = Matrix->Data + ElementIndex;
	*Element = Value;
}

bool MatricesAreEquivalent(matrix* M1, matrix* M2);
void SaveMatrix(matrix* Matrix, char* FilePath);
bool LoadMatrix(matrix* Matrix, char* FilePath);
#define MATRIX_H

#endif