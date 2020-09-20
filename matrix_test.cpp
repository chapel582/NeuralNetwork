#include "matrix.h"

bool IsBigEndian(void)
{
	uint32_t Value = 0xAABBCCDD;
	uint8_t* ValuePtr = (uint8_t*) &Value;
	return *ValuePtr == 0xAA;
}

void FillMatrixWithValue(matrix* Matrix, float Value)
{
	for(uint32_t Row = 0; Row < Matrix->NumRows; Row++)
	{
		for(uint32_t Col = 0; Col < Matrix->NumColumns; Col++)
		{
			SetMatrixElement(Matrix, Row, Col, Value);
		}
	}
}

void FillMatrixConsecutive(matrix* Matrix)
{
	// NOTE: test code for easily generating a matrix with some consecutive 
	// CONT: data
	for(uint32_t Row = 0; Row < Matrix->NumRows; Row++)
	{
		for(uint32_t Col = 0; Col < Matrix->NumColumns; Col++)
		{
			SetMatrixElement(
				Matrix, Row, Col, (float) (Matrix->NumColumns * Row + Col + 1)
			);
		}
	}
}

void FillMatrixNegativeConsecutive(matrix* Matrix)
{
	// NOTE: test code for easily generating a matrix with some consecutive 
	// CONT: data
	for(uint32_t Row = 0; Row < Matrix->NumRows; Row++)
	{
		for(uint32_t Col = 0; Col < Matrix->NumColumns; Col++)
		{
			SetMatrixElement(
				Matrix, Row, Col, -1.0f * (Matrix->NumColumns * Row + Col + 1)
			);
		}
	}
}

void FillOneHotMatrix(matrix* Matrix)
{
	MatrixClear(Matrix);
	for(uint32_t SampleIndex = 0; SampleIndex < Matrix->NumRows; SampleIndex++)
	{	
		SetMatrixElement(
			Matrix, SampleIndex, SampleIndex % Matrix->NumColumns, 1.0f
		);
	}
}