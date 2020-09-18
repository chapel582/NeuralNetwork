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

HOST_PREFIX DEVICE_PREFIX
void MatrixMultCore(
	matrix* M1, matrix* M2, matrix* Result, int Start, int Stride
)
{
	// NOTE: Function to be called within one of your threads 
	// CONT: (CPU or GPU doesn't matter)

	assert(M1->NumColumns == M2->NumRows);
	uint32_t CommonDim = M1->NumColumns;
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = GetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Row = ResultIndex / ResultColumns;
		uint32_t Column = ResultIndex % ResultColumns;
		float DotProduct = 0.0f;
		for(uint32_t DPIndex = 0; DPIndex < CommonDim; DPIndex++)
		{
			DotProduct += (
				GetMatrixElement(M1, Row, DPIndex) * 
				GetMatrixElement(M2, DPIndex, Column)
			);
		}
		SetMatrixElement(Result, Row, Column, DotProduct);
	}
}

HOST_PREFIX DEVICE_PREFIX
void MatrixMultM1TransposeCore(
	matrix* M1, matrix* M2, matrix* Result, int Start, int Stride
)
{
	// NOTE: Function to be called within one of your threads 
	// CONT: (CPU or GPU doesn't matter)

	// NOTE: For transpose multiplication without allocating and initializing
	// CONT: a new matrix
	// NOTE: the number of rows in M1 should equal the number of rows in M2
	
	uint32_t CommonDim = M1->NumRows;
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = GetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Row = ResultIndex / ResultColumns;
		uint32_t Column = ResultIndex % ResultColumns;
		float DotProduct = 0.0f;
		for(uint32_t DPIndex = 0; DPIndex < CommonDim; DPIndex++)
		{
			DotProduct += (
				GetMatrixElement(M1, DPIndex, Row) * 
				GetMatrixElement(M2, DPIndex, Column)
			);
		}
		SetMatrixElement(Result, Row, Column, DotProduct);
	}
}

HOST_PREFIX DEVICE_PREFIX
void MatrixMultM2TransposeCore(
	matrix* M1, matrix* M2, matrix* Result, int Start, int Stride
)
{
	uint32_t CommonDim = M1->NumColumns;
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = GetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Row = ResultIndex / ResultColumns;
		uint32_t Column = ResultIndex % ResultColumns;
		float DotProduct = 0.0f;
		for(uint32_t DPIndex = 0; DPIndex < CommonDim; DPIndex++)
		{
			DotProduct += (
				GetMatrixElement(M1, Row, DPIndex) * 
				GetMatrixElement(M2, Column, DPIndex)
			);
		}
		SetMatrixElement(Result, Row, Column, DotProduct);
	}
}

HOST_PREFIX DEVICE_PREFIX
void MatrixMultM1M2TransposeCore(
	matrix* M1, matrix* M2, matrix* Result, int Start, int Stride
)
{
	uint32_t CommonDim = M1->NumRows;
	uint32_t ResultColumns = Result->NumColumns;
	uint32_t NumResultElements = GetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		uint32_t Row = ResultIndex / ResultColumns;
		uint32_t Column = ResultIndex % ResultColumns;
		float DotProduct = 0.0f;
		for(uint32_t DPIndex = 0; DPIndex < CommonDim; DPIndex++)
		{
			DotProduct += (
				GetMatrixElement(M1, DPIndex, Row) * 
				GetMatrixElement(M2, Column, DPIndex)
			);
		}
		SetMatrixElement(Result, Row, Column, DotProduct);
	}
}

HOST_PREFIX DEVICE_PREFIX
void MatrixAddCore(
	matrix* M1, matrix* M2, matrix* Result, int Start, int Stride
)
{
	uint32_t NumResultElements = GetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		SetMatrixElement(
			Result,
			ResultIndex,
			GetMatrixElement(M1, ResultIndex) + 
			GetMatrixElement(M2, ResultIndex)
		);
	}
}

HOST_PREFIX DEVICE_PREFIX
void MatrixSubtractCore(
	matrix* M1, matrix* M2, matrix* Result, int Start, int Stride
)
{
	assert(M1->NumRows == M2->NumRows);
	assert(M1->NumColumns == M2->NumColumns);

	uint32_t NumResultElements = GetMatrixArrayCount(Result);
	for(
		uint32_t ResultIndex = Start;
		ResultIndex < NumResultElements;
		ResultIndex += Stride
	)
	{
		SetMatrixElement(
			Result,
			ResultIndex,
			GetMatrixElement(M1, ResultIndex) -
			GetMatrixElement(M2, ResultIndex)
		);
	}
}