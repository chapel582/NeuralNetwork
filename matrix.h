#ifndef MATRIX_H

#include "project_flags.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

struct matrix
{
	uint32_t NumRows;
	uint32_t NumColumns;
	float* Data;
};

HOST_PREFIX DEVICE_PREFIX uint32_t GetMatrixArrayCount(matrix* Matrix);
HOST_PREFIX DEVICE_PREFIX size_t GetMatrixDataSize(matrix* Matrix);
HOST_PREFIX DEVICE_PREFIX float* GetMatrixRow(matrix* Matrix, uint32_t Row);
HOST_PREFIX DEVICE_PREFIX float GetMatrixElement(
	matrix* Matrix, uint32_t Row, uint32_t Column
);
HOST_PREFIX DEVICE_PREFIX float GetMatrixElement(
	matrix* Matrix, uint32_t ElementIndex
);
HOST_PREFIX DEVICE_PREFIX void SetMatrixElement(
	matrix* Matrix, uint32_t Row, uint32_t Column, float Value
);
HOST_PREFIX DEVICE_PREFIX void SetMatrixElement(
	matrix* Matrix, uint32_t ElementIndex, float Value
);

bool MatricesAreEquivalent(matrix* M1, matrix* M2);
void SaveMatrix(matrix* Matrix, char* FilePath);
bool LoadMatrix(matrix* Matrix, char* FilePath);
void WriteMatrix(matrix* Matrix, FILE* File);

HOST_PREFIX DEVICE_PREFIX
void MatrixMultCore(
	matrix* M1, matrix* M2, matrix* Result, int Start, int Stride
);

#define MATRIX_H

#endif