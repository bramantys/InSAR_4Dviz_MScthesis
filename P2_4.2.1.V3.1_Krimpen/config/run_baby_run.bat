@echo off
setlocal EnableExtensions

set "CONFIG_DIR=%~dp0"
for %%I in ("%CONFIG_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "PREFLIGHT=%PROJECT_ROOT%\_internal\pipeline\user_preflight.py"
set "RUNNER=%PROJECT_ROOT%\_internal\pipeline\run_pipeline.py"
set "PYTHONUTF8=1"

call :select_python

echo.
echo ================================================================================
echo  Proto2 Parcel Viewer Pipeline
echo ================================================================================
echo  Project root : %PROJECT_ROOT%
if "%PYTHON_CMD%"=="" (
  echo  Python       : NOT FOUND
) else (
  echo  Python       : %PYTHON_CMD%
  echo  Source       : %PYTHON_SOURCE%
)
echo ================================================================================

if "%PYTHON_CMD%"=="" (
  echo.
  echo [ERROR] Python was not found.
  echo Install Anaconda/Miniconda or Python, then run this file again.
  echo.
  pause
  exit /b 1
)

if not exist "%PREFLIGHT%" (
  echo [ERROR] Missing pipeline preflight: %PREFLIGHT%
  pause
  exit /b 1
)
if not exist "%RUNNER%" (
  echo [ERROR] Missing pipeline runner: %RUNNER%
  pause
  exit /b 1
)

echo.
pushd "%PROJECT_ROOT%"
"%PYTHON_CMD%" "%PREFLIGHT%"
set "CHECK_CODE=%ERRORLEVEL%"
popd
if not "%CHECK_CODE%"=="0" (
  echo.
  echo Preflight failed. No pipeline products were generated.
  echo Fix config\project_config.json or the files in data\, then try again.
  echo.
  pause
  exit /b %CHECK_CODE%
)

echo.
echo The configuration and input paths passed the quick check.
choice /C YN /N /M "Run the full pipeline now? [Y/N]: "
if errorlevel 2 (
  echo.
  echo Pipeline cancelled. Nothing was changed.
  exit /b 0
)

echo.
echo ================================================================================
echo  Running full pipeline
 echo ================================================================================
echo.
pushd "%PROJECT_ROOT%"
"%PYTHON_CMD%" "%RUNNER%"
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
echo ================================================================================
if "%EXIT_CODE%"=="0" (
  echo  PIPELINE FINISHED SUCCESSFULLY
  echo.
  echo  Open: %PROJECT_ROOT%\viz2_dev_v11.html
  echo  Run records were written to: %PROJECT_ROOT%\run_records
) else (
  echo  PIPELINE FAILED ^(error code %EXIT_CODE%^)
  echo.
  echo  Send the newest JSON and TXT files from run_records\ for diagnosis.
)
echo ================================================================================
echo.
pause
exit /b %EXIT_CODE%

:select_python
set "PYTHON_CMD="
set "PYTHON_SOURCE="
if not "%PROTO2_PYTHON%"=="" (
  set "PYTHON_CMD=%PROTO2_PYTHON%"
  set "PYTHON_SOURCE=PROTO2_PYTHON"
  exit /b 0
)
if not "%CONDA_PREFIX%"=="" if exist "%CONDA_PREFIX%\python.exe" (
  set "PYTHON_CMD=%CONDA_PREFIX%\python.exe"
  set "PYTHON_SOURCE=active conda environment"
  exit /b 0
)
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
  set "PYTHON_CMD=%PROJECT_ROOT%\.venv\Scripts\python.exe"
  set "PYTHON_SOURCE=project .venv"
  exit /b 0
)
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
  set "PYTHON_CMD=%PROJECT_ROOT%\venv\Scripts\python.exe"
  set "PYTHON_SOURCE=project venv"
  exit /b 0
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
  set "PYTHON_CMD=%USERPROFILE%\miniconda3\python.exe"
  set "PYTHON_SOURCE=user Miniconda"
  exit /b 0
)
if exist "%USERPROFILE%\anaconda3\python.exe" (
  set "PYTHON_CMD=%USERPROFILE%\anaconda3\python.exe"
  set "PYTHON_SOURCE=user Anaconda"
  exit /b 0
)
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_CMD (
  set "PYTHON_CMD=%%P"
  set "PYTHON_SOURCE=PATH"
)
exit /b 0
