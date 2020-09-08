/*
TODO: This is not a final platform layer
	- Fullscreen support
	- Non-job threading
	- sleep
	- control cursor visibility
	- Hardware acceleration
	- Blit speed improvements
	- Raw input and support for multiple keyboards
*/

// NOTE: Apocalypse stuff
#include "apocalypse.cpp"
#include "apocalypse_platform.h"

// NOTE: C stuff
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// NOTE: Windows stuff
#include <windows.h>
#include <Winuser.h>

// NOTE: Win32 Apocalypse stuff
#include "win32_apocalypse.h"

bool GlobalRunning = false;
win32_offscreen_buffer GlobalBackBuffer = {};

struct win32_offscreen_buffer
{
	uint32_t Width;
	uint32_t Height;
	int Pitch;
	int BytesPerPixel;
	void* Memory;
	BITMAPINFO Info;
};

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
			win32_window_dimension Dim = Win32CalculateWindowDimensions();
			MINMAXINFO* Mmi = (MINMAXINFO*) LParam;
			Mmi->ptMinTrackSize.x = Dim.Width;
			Mmi->ptMinTrackSize.y = Dim.Height;
			Mmi->ptMaxTrackSize.x = Dim.Width;
			Mmi->ptMaxTrackSize.y = Dim.Height;
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
	// NOTE: get memory for backbuffer
	{
		GlobalBackBuffer.Width = MNIST_DIM;
		GlobalBackBuffer.Height = MNIST_DIM;
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

		bool MouseDown = false;
		if(WindowHandle)
		{
			// TODO: query this on Windows
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
			int GameUpdateHz = MonitorRefreshHz / 2;
			float TargetSecondsPerFrame = 1.0f / (float) GameUpdateHz;

			matrix* DigitMatrix;
			AllocMatrix(&DigitMatrix, 1, MNIST_DIM * MNIST_DIM);
			MatrixClear(DigitMatrix);

			// NOTE: Since we specified CS_OWNDC, we can just grab this 
			// CONT: once and use it forever. No sharing
			HDC DeviceContext = GetDC(WindowHandle);

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
							uint16_t XPos = LParam & 0xFFFF;
							uint16_t YPos = ScreenHeight - ((LParam & 0xFFFF0000) >> 16);
							MouseDown = true;
							break;
						}
						case(WM_LBUTTONUP):
						{
							break;
						}
						case(WM_MOUSEMOVE):
						{
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
							switch(KeyboardEvent->Code)
							{
								case(0x0D): // NOTE: Return/Enter V-code
								{
									if(!WasDown && IsDown)
									{
										uint32_t* Pixel = GlobalBackBuffer.Memory;
										for(int X = 0; X < GlobalBackBuffer.Width; X++)
										{
											for(int Y = 0; Y < GlobalBackBuffer.Height; Y++)
											{
												if(*Pixel != 0)
												{
													SetMatrix();
												}

												*Pixel = 0;
												Pixel++;
											}
										}

										uint32_t* Pixel = GlobalBackBuffer.Memory;
										for(int X = 0; X < GlobalBackBuffer.Width; X++)
										{
											for(int Y = 0; Y < GlobalBackBuffer.Height; Y++)
											{
												*Pixel = 0;
												Pixel++;
											}
										}
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