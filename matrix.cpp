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