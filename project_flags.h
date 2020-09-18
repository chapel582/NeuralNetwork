#ifndef PROJECT_FLAGS_H

#ifdef __NVCC__
#define HOST_PREFIX __host__
#define DEVICE_PREFIX __device__
#else
#define HOST_PREFIX
#define DEVICE_PREFIX
#endif

#define PROJECT_FLAGS_H
#endif