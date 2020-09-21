#ifndef PERFORMANCE_H

int64_t Win32GetWallClock(void);
float Win32GetSecondsElapsed(int64_t Start, int64_t End);

#define PERFORMANCE_H
#endif