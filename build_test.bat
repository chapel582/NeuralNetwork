@echo off

REM TODO - can we just build both with one exe?

call .\setup.bat
call .\build_cpu.bat

set BuildFolder=.\build
set CommonCompilerFlags=-EHa- -Oi -Od -W4 -FC -Z7 -w44062
set CommonLinkerFlags=-opt:ref

IF NOT EXIST %BuildFolder% mkdir %BuildFolder%
pushd %BuildFolder%
del *.pdb > nul 2> nul
rem NOTE: Not using WX because I want to keep the 4100 warning but it will
rem CONT: always be hit on the WinMain callback 
rem NOTE: -w44062 is here because, for some reason, it is not seen by the 
rem CONT: level 4 warnings
rem NOTE: -EHA- Disabled exception handling
rem NOTE: -GR- Disabled runtime type information
rem NOTE: -MT is used so that we don't use the c-runtime library DLL but 
rem CONT: statically link instead 
rem NOTE: -Od should be removed for non-debug builds
cl %CommonCompilerFlags% ..\test.cpp -Fmtest.map /link %CommonLinkerFlags% 
popd