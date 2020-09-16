@echo off

REM TODO - can we just build both with one exe?

call .\setup.bat

set BuildFolder=.\build
IF NOT EXIST %BuildFolder% mkdir %BuildFolder%
pushd %BuildFolder%

rem TODO: add option to remove debug compilation
rem TODO: build nerual_net_cuda.cu as DLL
nvcc ..\neural_net_cuda_test.cu -o neural_net_cuda_test.exe --debug --device-debug -l Shlwapi
nvcc ..\neural_net_cuda_performance.cu -o neural_net_cuda_performance.exe --debug --device-debug

popd