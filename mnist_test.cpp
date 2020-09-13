#include "mnist_test.h"

#include "matrix.h"

#include <stdint.h>

int LoadMnistDigitCsv(
	matrix* Data, matrix* Labels, uint32_t MnistTrainSamples, char* FilePath
)
{
	// NOTE: initializes normalized data and label matrices corresponding to 
	// CONT: MNIST handwriting database
	char ReadBuffer[8];

	FILE* File;
	fopen_s(&File, FilePath, "r");
	if(File == NULL)
	{
		goto error;
	}

	for(
		uint32_t SampleIndex = 0;
		SampleIndex < MnistTrainSamples;
		SampleIndex++
	)
	{
		// NOTE: read label
		fread(ReadBuffer, 1, 1, File);
		int Label = atoi(ReadBuffer);
		// NOTE: b/c we cleared labels matrix, only need to set 1.0f value
		SetMatrixElement(Labels, SampleIndex, Label, 1.0f);

		// NOTE: read comma
		fread(ReadBuffer, 1, 1, File);

		for(uint32_t DataIndex = 0; DataIndex < MNIST_DATA_SIZE; DataIndex++)
		{
			char* ReadTo = ReadBuffer;
			while(true)
			{
				fread(ReadTo, 1, 1, File);
				if((*ReadTo == ',') || (*ReadTo == '\n'))
				{
					*ReadTo = 0;
					break;
				}
				else
				{
					ReadTo++;
				}
			}

			int PixelValue = atoi(ReadBuffer);
			SetMatrixElement(
				Data,
				SampleIndex,
				DataIndex,
				((float) PixelValue) / MNIST_MAX_PIXEL_VALUE
			);
		}
	}
	fclose(File);
	goto end;

error:
	return 1;
end:
	return 0;
}