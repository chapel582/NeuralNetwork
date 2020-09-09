#if DEMO_DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include "neural_net_cpu.cpp"
#include "vector.h"

// NOTE: C stuff
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// NOTE: Windows stuff
#include <windows.h>
#include <Winuser.h>

bool GlobalRunning = false;

struct win32_offscreen_buffer
{
	uint32_t Width;
	uint32_t Height;
	int Pitch;
	int BytesPerPixel;
	void* Memory;
	BITMAPINFO Info;
};
win32_offscreen_buffer GlobalBackBuffer = {};

#define WINDOW_STYLE (WS_OVERLAPPEDWINDOW | WS_VISIBLE)

// START SECTION: Performance counters
int64_t GlobalPerformanceFrequency = 0;
inline int64_t Win32GetWallClock(void)
{
	LARGE_INTEGER Result;
	// NOTE: QueryPerformanceCounter gets wall clock time
	QueryPerformanceCounter(&Result);
	return Result.QuadPart;
}

inline float Win32GetSecondsElapsed(int64_t Start, int64_t End)
{
	float Result;
	Result = (
		((float) (End - Start)) / 
		((float) GlobalPerformanceFrequency)
	);
	return Result;
}
// STOP SECTION: Performance counters

void Win32BufferToWindow(win32_offscreen_buffer* BackBuffer, HDC DeviceContext)
{
	StretchDIBits(
		DeviceContext,
		0,
		0,
		BackBuffer->Width,
		BackBuffer->Height,
		0,
		0,
		BackBuffer->Width,
		BackBuffer->Height,
		BackBuffer->Memory,
		&BackBuffer->Info,
		DIB_RGB_COLORS,
		SRCCOPY
	);
}

struct win32_window_dimension
{
	uint32_t Width;
	uint32_t Height;
};

win32_window_dimension Win32GetWindowDimension(HWND Window)
{
	RECT ClientRect = {};
	GetClientRect(Window, &ClientRect);
	win32_window_dimension Result = {};
	Result.Width = ClientRect.right - ClientRect.left;
	Result.Height = ClientRect.bottom - ClientRect.top;
	return Result;
}

win32_window_dimension Win32CalculateWindowDimensions()
{
	RECT ClientRect = {};
	ClientRect.right = GlobalBackBuffer.Width;
	ClientRect.bottom = GlobalBackBuffer.Height;
	AdjustWindowRect(
		&ClientRect,
		WINDOW_STYLE,
		false
	);
	win32_window_dimension Result;
	Result.Width = ClientRect.right - ClientRect.left;
	Result.Height = ClientRect.bottom - ClientRect.top;
	return Result;
}

LRESULT CALLBACK MainWindowCallback(
	HWND Window,
	UINT Message,
	WPARAM WParam,
	LPARAM LParam
)
{
	LRESULT Result = 0;
	switch(Message)
	{
		case(WM_SIZE):
		{
			break;
		}
		case(WM_ACTIVATEAPP):
		{
			break;
		}
		case(WM_CLOSE):
		{
			GlobalRunning = false;
			break;
		}
		case(WM_DESTROY):
		{
			GlobalRunning = false;
			break;
		}
		case(WM_SYSKEYDOWN):
		case(WM_SYSKEYUP):
		case(WM_KEYDOWN):
		case(WM_KEYUP):
		{
			break;
		}
		case(WM_LBUTTONDOWN):
		case(WM_LBUTTONUP):
		case(WM_RBUTTONDOWN):
		case(WM_RBUTTONUP):
		case(WM_MOUSEMOVE):
		{
			break;
		}
		case(WM_PAINT):
		{
			PAINTSTRUCT Paint = {};
			HDC DeviceContext = BeginPaint(Window, &Paint);
			
			Win32BufferToWindow(&GlobalBackBuffer, DeviceContext);

			EndPaint(Window, &Paint);
			break;
		}
		case(WM_GETMINMAXINFO):
		{
			break;
		}
		default:
		{
			Result = DefWindowProc(Window, Message, WParam, LParam);
			break;
		}
	}

	return Result;
}

#define MNIST_DIM 28

int CALLBACK WinMain(
	HINSTANCE Instance,
	HINSTANCE PrevInstance,
	LPSTR CommandLine, 
	int ShowCode
)
{
#if DEMO_DEBUG
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	bool ConsoleWorking = AllocConsole();
	if(!ConsoleWorking)
	{
		goto end;
	}
	FILE* Stream;
	freopen_s(&Stream, "CONOUT$", "w", stdout);

	LARGE_INTEGER PerformanceFrequency;
	QueryPerformanceFrequency(&PerformanceFrequency);
	GlobalPerformanceFrequency = PerformanceFrequency.QuadPart;

	// NOTE: set the windows scheduler granularity to 1 ms so that our Sleep()
	// CONT: can be more granular
	UINT DesiredSchedulerMS = 1;
	bool SleepIsGranular = (
		timeBeginPeriod(DesiredSchedulerMS) == TIMERR_NOERROR
	);

	GlobalBackBuffer = {};
	GlobalBackBuffer.BytesPerPixel = 4;
	uint32_t ImageScaleUp = 16;
	uint32_t BrushWidth = 4 * ImageScaleUp;
	// NOTE: get memory for backbuffer
	{
		GlobalBackBuffer.Width = ImageScaleUp * MNIST_DIM;
		GlobalBackBuffer.Height = ImageScaleUp * MNIST_DIM;
		GlobalBackBuffer.Pitch = (
			GlobalBackBuffer.Width * GlobalBackBuffer.BytesPerPixel
		);

		GlobalBackBuffer.Info.bmiHeader.biSize = (
			sizeof(GlobalBackBuffer.Info.bmiHeader)
		);
		GlobalBackBuffer.Info.bmiHeader.biWidth = GlobalBackBuffer.Width;
		GlobalBackBuffer.Info.bmiHeader.biHeight = GlobalBackBuffer.Height;
		GlobalBackBuffer.Info.bmiHeader.biPlanes = 1;
		GlobalBackBuffer.Info.bmiHeader.biBitCount = 32;
		GlobalBackBuffer.Info.bmiHeader.biCompression = BI_RGB;

		size_t BitmapMemorySize = (
			(GlobalBackBuffer.Width * GlobalBackBuffer.Height) * 
			GlobalBackBuffer.BytesPerPixel
		);
		if(GlobalBackBuffer.Memory)
		{
			VirtualFree(GlobalBackBuffer.Memory, 0, MEM_RELEASE);
		}
		GlobalBackBuffer.Memory = VirtualAlloc(
			0, BitmapMemorySize, MEM_COMMIT, PAGE_READWRITE
		);
	}

	WNDCLASS WindowClass = {};
	WindowClass.lpfnWndProc = MainWindowCallback;
	WindowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	WindowClass.hInstance = Instance;
	WindowClass.lpszClassName = "DemoWindowClass";

	if(RegisterClassA(&WindowClass))
	{
		win32_window_dimension WindowDim = Win32CalculateWindowDimensions();
		HWND WindowHandle = CreateWindowExA(
			0,
			WindowClass.lpszClassName,
			"DemoDigitClassifier",
			WINDOW_STYLE,
			0,
			0,
			WindowDim.Width,
			WindowDim.Height,
			0,
			0,
			Instance,
			0
		);

		if(WindowHandle)
		{
			int MonitorRefreshHz = 60;
			{
				HDC RefreshDC = GetDC(WindowHandle);
				int Win32RefreshRate = GetDeviceCaps(RefreshDC, VREFRESH);
				ReleaseDC(WindowHandle, RefreshDC);
				if(Win32RefreshRate > 1)
				{
					MonitorRefreshHz = Win32RefreshRate;
				}
			}
			int UpdateHz = MonitorRefreshHz / 2;
			float TargetSecondsPerFrame = 1.0f / (float) UpdateHz;

			matrix* PredictMatrix;
			AllocMatrix(&PredictMatrix, 1, MNIST_DIM * MNIST_DIM);
			MatrixClear(PredictMatrix);

			// NOTE: Since we specified CS_OWNDC, we can just grab this 
			// CONT: once and use it forever. No sharing
			HDC DeviceContext = GetDC(WindowHandle);

			
			char NeuralNetModelPath[260];
			if(CommandLine[0] == 0)
			{
				strcpy_s(
					NeuralNetModelPath,					
					sizeof(NeuralNetModelPath),
					"../test_data/models/mnist_16384samples.model"		
				);
			}
			else
			{
				strcpy_s(
					NeuralNetModelPath,
					sizeof(NeuralNetModelPath),
					CommandLine
				);
			}
			printf("Testing %s\n", NeuralNetModelPath);

			uint32_t MiniBatchSize = 32;
			matrix* TrainMatrix;
			AllocMatrix(&TrainMatrix, MiniBatchSize, MNIST_DIM * MNIST_DIM);
			MatrixClear(TrainMatrix);
			matrix* TrainLabels;
			AllocMatrix(&TrainLabels, MiniBatchSize, 10);
			MatrixClear(TrainLabels);

			uint32_t TrainDataIndex = 0;
			neural_net* TrainingNn = NULL;
			LoadNeuralNet(
				&TrainingNn, NeuralNetModelPath, MiniBatchSize, 4
			);
			neural_net* PredictionNn;
			ResizedNeuralNet(&PredictionNn, TrainingNn, 1);

			neural_net_trainer* Trainer;
			AllocNeuralNetTrainer(
				&Trainer,
				TrainingNn,
				0.1f,
				LayerType_Count
			);

			bool MouseDown = false;
			uint16_t LastY = 0;
			uint16_t LastX = 0;

			int64_t FlipWallClock = Win32GetWallClock();
			GlobalRunning = true;
			while(GlobalRunning)
			{
				int64_t FrameStartCounter = Win32GetWallClock();
				// NOTE: rdtsc gets cycle counts instead of wall clock time
				uint64_t FrameStartCycle = __rdtsc();

				MSG Message = {};
				while(PeekMessage(&Message, 0, 0, 0, PM_REMOVE))
				{
					if(Message.message == WM_QUIT)
					{
						GlobalRunning = false;
					}
					switch(Message.message)
					{
						case(WM_LBUTTONDOWN):
						{
							uint16_t XPos = Message.lParam & 0xFFFF;
							uint16_t YPos = (uint16_t) (
								GlobalBackBuffer.Height - 
								((Message.lParam & 0xFFFF0000) >> 16)
							);
							LastY = YPos;
							LastX = XPos;
							break;
						}
						case(WM_LBUTTONUP):
						{
							break;
						}
						case(WM_MOUSEMOVE):
						{
							if(((Message.wParam & MK_LBUTTON) > 0))
							{
								uint16_t XPos = Message.lParam & 0xFFFF;
								uint16_t YPos = (uint16_t)(
									GlobalBackBuffer.Height - 
									((Message.lParam & 0xFFFF0000) >> 16)
								);

								uint32_t* Pixel = (uint32_t*) (
									GlobalBackBuffer.Memory
								);
								Pixel += YPos * GlobalBackBuffer.Width + XPos;
								*Pixel = 0xFFFFFFFF;

								vector2 NewPos = Vector2(XPos, YPos);
								vector2 LastPos = Vector2(LastX, LastY);
								vector2 Diff = NewPos - LastPos;
								vector2 NormalDiff = Normalize(Diff);
								vector2 CurrentPixel = LastPos;
								while(Magnitude(CurrentPixel - NewPos) > 1)
								{
									uint32_t IntPixelX = (uint32_t) (
										CurrentPixel.X
									);
									uint32_t IntPixelY = (uint32_t) (
										CurrentPixel.Y
									);

									// NOTE: here we make our brush bigger
									uint32_t Left = (
										IntPixelX - (BrushWidth / 2)
									);
									uint32_t Top = (
										IntPixelY - (BrushWidth / 2)
									);
									for(
										uint32_t RectY = Top;
										RectY < (Top + BrushWidth);
										RectY++
									)
									{
										for(
											uint32_t RectX = Left;
											RectX < (Left + BrushWidth);
											RectX++
										)
										{
											if(
												RectX >= 0 &&
												RectX < GlobalBackBuffer.Width &&
												RectY >= 0 &&
												RectY < GlobalBackBuffer.Height
											)
											{
												uint32_t* WriteTo = (
													((uint32_t*) GlobalBackBuffer.Memory) +
													GlobalBackBuffer.Width * RectY + 
													RectX
												);
												*WriteTo = 0xFFFFFFFF;
											}
										}
									}

									CurrentPixel += NormalDiff;
								}
								LastY = YPos;
								LastX = XPos;
							}
							break;
						}
						case(WM_SYSKEYDOWN):
						case(WM_SYSKEYUP):
						case(WM_KEYDOWN):
						case(WM_KEYUP):
						{
							uint8_t Code = (uint8_t) Message.wParam;
							bool WasDown = (
								(Message.lParam & (1 << 30)) != 0
							);
							bool IsDown = (
								(Message.lParam & (1 << 31)) == 0
							);
							switch((uint8_t) Message.wParam)
							{
								case(0x20):
								{
									// NOTE: save model
									if(!WasDown && IsDown)
									{
										char FilePathBuffer[260];
										snprintf(
											FilePathBuffer,
											sizeof(FilePathBuffer),
											"%strain",
											NeuralNetModelPath
										);
										printf(
											"Saving model to %s\n", FilePathBuffer
										);
										SaveNeuralNet(TrainingNn, FilePathBuffer);
										break;
									}									
								}
								case(0x30):
								case(0x31):
								case(0x32):
								case(0x33):
								case(0x34):
								case(0x35):
								case(0x36):
								case(0x37):
								case(0x38):
								case(0x39):								
								{
									if(!WasDown && IsDown)
									{
										// NOTE: num keys
										uint32_t* Pixel = (uint32_t*) (
											GlobalBackBuffer.Memory
										);
										for(
											uint32_t Y = 0;
											Y < GlobalBackBuffer.Height;
											Y++
										)
										{
											for(
												uint32_t X = 0;
												X < GlobalBackBuffer.Width;
												X++
											)
											{
												if(*Pixel != 0)
												{
													uint32_t FlippedY = (
														GlobalBackBuffer.Height - 
														Y
													);

													uint32_t DownSampleX = (
														X / ImageScaleUp
													);
													uint32_t DownSampleY = (
														FlippedY / ImageScaleUp
													);

													uint32_t MatrixIndex = (
														(MNIST_DIM * DownSampleY)
														+ 
														DownSampleX
													);

													float LastValue = (
														GetMatrixElement(
															TrainMatrix,
															TrainDataIndex,
															MatrixIndex
														)
													);
													SetMatrixElement(
														TrainMatrix,
														TrainDataIndex,
														MatrixIndex,
														LastValue + 1.0f
													);
												}

												*Pixel = 0;
												Pixel++;
											}
										}

										uint8_t Label = (
											(uint8_t) Message.wParam - 0x30
										);
										SetMatrixElement(
											TrainLabels,
											TrainDataIndex,
											Label,
											1.0f
										);

										if(
											TrainDataIndex == (MiniBatchSize - 1)
										)
										{
											printf(
												"Full minibatch. Training!\n"
											);
											MatrixScalarMultCore(
												(
													1.0f / 
													(
														(float) ImageScaleUp * 
														(float) ImageScaleUp
													)
												),
												TrainMatrix,
												TrainMatrix,
												0,
												1
											);
											TrainNeuralNet(
												Trainer,
												TrainingNn,
												TrainMatrix,
												TrainLabels,
												1,
												false,
												false
											);
											TrainDataIndex = 0;

											MatrixClear(TrainMatrix);
											MatrixClear(TrainLabels);
											printf("Training complete\n");
										}
										else
										{
											printf(
												"Sample %d. Saving data as %d...\n",
												TrainDataIndex,
												Label
											);
											TrainDataIndex++;
										}
									}
									break;
								}
								case(0x0D): // NOTE: Return/Enter V-code
								{
									if(!WasDown && IsDown)
									{
										MatrixClear(PredictMatrix);

										uint32_t* Pixel = (uint32_t*) (
											GlobalBackBuffer.Memory
										);
										for(
											uint32_t Y = 0;
											Y < GlobalBackBuffer.Height;
											Y++
										)
										{
											for(
												uint32_t X = 0;
												X < GlobalBackBuffer.Width;
												X++
											)
											{
												if(*Pixel != 0)
												{
													uint32_t FlippedY = (
														GlobalBackBuffer.Height - 
														Y
													);

													uint32_t DownSampleX = (
														X / ImageScaleUp
													);
													uint32_t DownSampleY = (
														FlippedY / ImageScaleUp
													);

													uint32_t MatrixIndex = (
														(MNIST_DIM * DownSampleY)
														+ 
														DownSampleX
													);

													float LastValue = (
														GetMatrixElement(
															PredictMatrix,
															0,
															MatrixIndex
														)
													);
													SetMatrixElement(
														PredictMatrix,
														0,
														MatrixIndex,
														LastValue + 1.0f
													);
												}

												*Pixel = 0;
												Pixel++;
											}
										}

										MatrixScalarMultCore(
											(
												1.0f / 
												(
													(float) ImageScaleUp * 
													(float) ImageScaleUp
												)
											),
											PredictMatrix,
											PredictMatrix,
											0,
											1
										);

										int Prediction = Predict(
											PredictionNn, PredictMatrix, 0
										);
										printf(
											"Predicted digit: %d\n", Prediction
										);
									}
									break;
								}
							}
							break;
						}
						default:
						{
							TranslateMessage(&Message);
							DispatchMessageA(&Message);
							break;
						}
					}
				}				

				uint64_t WorkEndCounter = Win32GetWallClock();
				float WorkSeconds = Win32GetSecondsElapsed(
					FrameStartCounter, WorkEndCounter
				);
				float SecondsElapsedForFrame = WorkSeconds;
				if(SecondsElapsedForFrame < TargetSecondsPerFrame)
				{
					if(SleepIsGranular)
					{
						// NOTE: casting down so we don't sleep too long
						uint32_t SleepMs = (uint32_t) (
							1000.0f * (
								TargetSecondsPerFrame - SecondsElapsedForFrame
							)
						);
						if(SleepMs > 0)
						{
							Sleep(SleepMs - 1);
						}
					}

					while(SecondsElapsedForFrame < TargetSecondsPerFrame)
					{
						SecondsElapsedForFrame = Win32GetSecondsElapsed(
							FrameStartCounter, Win32GetWallClock()
						);
					}
				}

				uint64_t FrameEndCycle = __rdtsc();
				int64_t FrameEndCounter = Win32GetWallClock();

				Win32BufferToWindow(&GlobalBackBuffer, DeviceContext);
				FlipWallClock = Win32GetWallClock();				
			}
		}
		else
		{
			// TODO: Put this in a more unified logging location and get more 
			// CONT: info
			OutputDebugStringA("Failed to get window handle\n");
			goto end;
		}
	}
	else
	{
		// TODO: Put this in a more unified logging location and get more info
		OutputDebugStringA("Failed to register window class\n");
		goto end;
	}

end:
	return 0;
}