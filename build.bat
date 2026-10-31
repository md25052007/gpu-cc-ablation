@echo off
REM Windows build. Run from "x64 Native Tools Command Prompt for VS" (so cl.exe is on PATH),
REM or from any prompt if nvcc already finds Visual Studio's compiler.
REM   build.bat            -> bin\cc.exe and bin\cc_counters.exe for the GPU in this PC
REM   build.bat sm_86      -> pick the architecture explicitly (needed for CUDA older than 11.5)
setlocal
set ARCH=%1
if "%ARCH%"=="" set ARCH=native
if not exist bin mkdir bin
echo Building timing build (arch=%ARCH%)...
nvcc -O3 -std=c++17 -arch=%ARCH% -lineinfo -o bin\cc.exe src\cc.cu || goto :fail
echo Building counter build...
nvcc -O3 -std=c++17 -arch=%ARCH% -lineinfo -DCC_COUNTERS -o bin\cc_counters.exe src\cc.cu || goto :fail
echo.
echo Build OK. Next:  python tools\run_tests.py
exit /b 0
:fail
echo.
echo BUILD FAILED. Check: nvcc --version works, Visual Studio C++ build tools installed,
echo and you are in the "x64 Native Tools Command Prompt for VS".
exit /b 1
