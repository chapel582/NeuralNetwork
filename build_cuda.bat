@echo off

REM TODO - can we just build both with one exe?

call .\setup.bat

set BuildFolder=.\build
IF NOT EXIST %BuildFolder% mkdir %BuildFolder%
pushd %BuildFolder%

rem TODO: add option to remove debug compilation
nvcc ..\neural_net_cuda.cu -o neural_net_cuda.exe --debug --device-debug

popd