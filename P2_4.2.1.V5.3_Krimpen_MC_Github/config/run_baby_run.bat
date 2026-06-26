@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Proto2 Krimpenerwaard - Parcel Viewer Build

rem ============================================================================
rem Proto2 portable launcher
rem
rem Selection policy:
rem   1. Explicit PROTO2_PYTHON override.
rem   2. The Python already active on PATH (same principle as Proto1 V7.1).
rem   3. A project-local .venv / venv.
rem   4. Active virtualenv / Conda environment paths.
rem   5. Generic Windows discovery: registered Pythons and all Conda envs.
rem
rem A candidate is accepted only when it can import every package required by
rem this project. The package guide is readme\requirements.txt. No machine-specific
rem environment names are used.
rem ============================================================================

set "CONFIG_DIR=%~dp0"
for %%I in ("%CONFIG_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "PIPELINE=%PROJECT_ROOT%\_internal\pipeline\run_full_pipeline.py"
set "PREFLIGHT=%PROJECT_ROOT%\_internal\pipeline\launcher_preflight.py"
rem The package keeps user-facing installation notes under readme/.
rem Root-level requirements.txt remains accepted for older copies.
set "REQUIREMENTS="
set "PYTHONUTF8=1"

:START
cls
call :resolve_requirements
call :select_python

echo.
echo ================================================================================
echo  Proto2 Krimpenerwaard Parcel Viewer
echo  data ^> config ^> automatic preflight ^> automatic pipeline ^> viewer
echo ================================================================================
echo  Project root : %PROJECT_ROOT%
echo  Viewer       : viz2_parcel_viewer.html
echo  Runtime      : _internal\data_pipeline\runtime
echo  Temporary    : cleaned automatically after a successful build
echo ================================================================================

if not defined PYTHON_CMD (
  echo.
  echo [ERROR] No compatible Python environment was found.
  echo.
  echo This project needs the packages listed in readme\requirements.txt:
  echo numpy, pandas, pyarrow, geopandas, shapely, pyproj, mapbox-earcut
  echo.
  echo Portable ways to provide one:
  echo   - Activate your existing Python or Conda environment, then run this BAT.
  echo   - Create a project-local .venv and install readme\requirements.txt into it.
  echo   - Set PROTO2_PYTHON to the full path of a suitable python.exe.
  goto :FAILED
)

if not exist "%PIPELINE%" (
  echo.
  echo [ERROR] Missing pipeline runner:
  echo %PIPELINE%
  goto :FAILED
)
if not exist "%PREFLIGHT%" (
  echo.
  echo [ERROR] Missing launcher preflight:
  echo %PREFLIGHT%
  goto :FAILED
)
if not defined REQUIREMENTS (
  echo.
  echo [ERROR] Requirements guide is missing.
  echo.
  echo Expected canonical location:
  echo %PROJECT_ROOT%\readme\requirements.txt
  echo.
  echo Compatibility fallback also accepted:
  echo %PROJECT_ROOT%\requirements.txt
  goto :FAILED
)
echo [OK] Requirements guide: %REQUIREMENTS%
if not exist "%PROJECT_ROOT%\_internal\cesium\Cesium.js" (
  echo.
  echo [ERROR] Cesium is missing.
  echo Copy your complete existing _internal\cesium folder into this template first.
  echo Expected: _internal\cesium\Cesium.js
  goto :FAILED
)
if not exist "%PROJECT_ROOT%\_internal\cesium\Widgets\widgets.css" (
  echo.
  echo [ERROR] Cesium Widgets CSS is missing.
  echo Copy the complete existing _internal\cesium folder into this template first.
  goto :FAILED
)
if not exist "%PROJECT_ROOT%\_internal\three\three.min.js" (
  echo.
  echo [ERROR] Local Three.js runtime is missing.
  echo Expected: _internal\three\three.min.js
  goto :FAILED
)

echo.
echo [CHECK] Selected Python environment
"%PYTHON_CMD%" -c "import sys; print('[OK] Python ' + sys.version.split()[0]); print('[OK] Executable: ' + sys.executable)"
if errorlevel 1 goto :FAILED
echo [OK] Selected by: %PYTHON_SOURCE%

echo.
echo [CHECK] Environment, configuration, and input contract
pushd "%PROJECT_ROOT%"
"%PYTHON_CMD%" "%PREFLIGHT%"
set "PREFLIGHT_CODE=%ERRORLEVEL%"
popd
if not "%PREFLIGHT_CODE%"=="0" goto :FAILED

echo.
echo [WORKING] Preflight passed. Starting the full pipeline automatically...
pushd "%PROJECT_ROOT%"
"%PYTHON_CMD%" "%PIPELINE%"
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
echo ================================================================================
if "%EXIT_CODE%"=="0" (
  echo  BUILD COMPLETE
  echo.
  echo  Serve this file with VS Code Live Server:
  echo  %PROJECT_ROOT%\viz2_parcel_viewer.html
  echo.
  echo  Receipt:
  echo  %PROJECT_ROOT%\run_records\latest_run.txt
  echo ================================================================================
  echo.
  choice /C RE /N /M "[R] Run again  [E] Exit: "
  if errorlevel 2 exit /b 0
  goto :START
) else (
  echo  BUILD FAILED ^(error code %EXIT_CODE%^)
  echo.
  echo  Read this exact report:
  echo  %PROJECT_ROOT%\run_records\latest_run.txt
  echo ================================================================================
  echo.
  choice /C RE /N /M "[R] Return to start  [E] Exit: "
  if errorlevel 2 exit /b %EXIT_CODE%
  goto :START
)

:FAILED
echo.
echo ================================================================================
echo  SETUP OR PREFLIGHT FAILED
echo  Fix the displayed error, then run this BAT again.
echo ================================================================================
echo.
choice /C RE /N /M "[R] Return to start  [E] Exit: "
if errorlevel 2 exit /b 1
goto :START

:resolve_requirements
set "REQUIREMENTS="
if exist "%PROJECT_ROOT%\readme\requirements.txt" (
  set "REQUIREMENTS=%PROJECT_ROOT%\readme\requirements.txt"
  exit /b 0
)
if exist "%PROJECT_ROOT%\requirements.txt" (
  set "REQUIREMENTS=%PROJECT_ROOT%\requirements.txt"
)
exit /b 0

:select_python
set "PYTHON_CMD="
set "PYTHON_SOURCE="
set "PYTHON_ATTEMPT_FILE=%TEMP%\proto2_python_attempts_%RANDOM%_%RANDOM%.txt"
type nul > "%PYTHON_ATTEMPT_FILE%"

echo.
echo [SEEK] Looking for a compatible Python environment...

rem 1. Explicit override for any user who wants to select an interpreter.
if defined PROTO2_PYTHON (
  set "PROTO2_PYTHON_CANDIDATE=%PROTO2_PYTHON:"=%"
  call :try_python "%PROTO2_PYTHON_CANDIDATE%" "PROTO2_PYTHON override"
  if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
)

rem 2. Proto1 V7.1 behaviour first: use Python already chosen by the user/PATH.
for /f "delims=" %%P in ('where python 2^>nul') do (
  call :try_python "%%P" "Python on PATH (Proto1-compatible)"
  if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
)

rem 3. A self-contained project virtual environment is portable and preferred
rem    after an intentionally active PATH environment.
call :try_python "%PROJECT_ROOT%\.venv\Scripts\python.exe" "project .venv"
if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
call :try_python "%PROJECT_ROOT%\venv\Scripts\python.exe" "project venv"
if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE

rem 4. Explicit active environment locations, in case PATH was not updated.
if defined VIRTUAL_ENV (
  call :try_python "%VIRTUAL_ENV%\Scripts\python.exe" "active virtual environment"
  if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
)
if defined CONDA_PREFIX (
  call :try_python "%CONDA_PREFIX%\python.exe" "active Conda environment"
  if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
)

rem 5. Generic common Conda roots. These are standard Windows locations, not
rem    a machine-specific environment name.
call :scan_conda_root "%USERPROFILE%\miniconda3"
if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
call :scan_conda_root "%USERPROFILE%\anaconda3"
if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
call :scan_conda_root "%LOCALAPPDATA%\miniconda3"
if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
call :scan_conda_root "%LOCALAPPDATA%\anaconda3"
if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
call :scan_conda_root "%PROGRAMDATA%\miniconda3"
if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE
call :scan_conda_root "%PROGRAMDATA%\anaconda3"
if defined PYTHON_CMD goto :PYTHON_SELECTION_DONE

rem 6. Generic Windows discovery. This searches Python Launcher registrations and
rem    all environments reported by whatever Conda installation is available.
call :discover_generic_python

:PYTHON_SELECTION_DONE
if exist "%PYTHON_ATTEMPT_FILE%" del /q "%PYTHON_ATTEMPT_FILE%" >nul 2>&1
exit /b 0

:try_python
set "CANDIDATE_PYTHON=%~1"
set "CANDIDATE_SOURCE=%~2"
if "%CANDIDATE_PYTHON%"=="" exit /b 1
if not exist "%CANDIDATE_PYTHON%" exit /b 1
if exist "%PYTHON_ATTEMPT_FILE%" findstr /L /X /C:"%CANDIDATE_PYTHON%" "%PYTHON_ATTEMPT_FILE%" >nul 2>&1 && exit /b 1
if exist "%PYTHON_ATTEMPT_FILE%" >> "%PYTHON_ATTEMPT_FILE%" echo %CANDIDATE_PYTHON%

echo [TRY] %CANDIDATE_SOURCE%
"%CANDIDATE_PYTHON%" -c "import sys; import numpy, pandas, pyarrow, geopandas, shapely, pyproj, mapbox_earcut; print(sys.executable)" >nul 2>&1
if errorlevel 1 (
  echo       skipped: required project packages are unavailable or cannot import.
  exit /b 1
)

set "PYTHON_CMD=%CANDIDATE_PYTHON%"
set "PYTHON_SOURCE=%CANDIDATE_SOURCE%"
echo [SELECTED] %CANDIDATE_SOURCE%
exit /b 0

:discover_generic_python
where powershell >nul 2>&1
if errorlevel 1 (
  echo [INFO] Windows generic discovery skipped: PowerShell is unavailable.
  exit /b 0
)

set "DISCOVERY_FILE=%TEMP%\proto2_python_candidates_%RANDOM%_%RANDOM%.txt"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $paths=@(); Get-Command python -All | ForEach-Object { if($_.Path){ $paths += $_.Path } elseif($_.Source){ $paths += $_.Source } }; if(Get-Command py -ErrorAction SilentlyContinue){ try { (& py -0p) | ForEach-Object { if($_ -match '([A-Za-z]:\\.*?python\.exe)\s*$'){ $paths += $Matches[1] } } } catch {} }; if(Get-Command conda -ErrorAction SilentlyContinue){ try { $envs = ((& conda env list --json | Out-String | ConvertFrom-Json).envs); $envs | ForEach-Object { $paths += (Join-Path $_ 'python.exe') } } catch {} }; $paths | ForEach-Object { if($_ -and (Test-Path -LiteralPath $_)){ [IO.Path]::GetFullPath($_) } } | Sort-Object -Unique" > "%DISCOVERY_FILE%" 2>nul

if exist "%DISCOVERY_FILE%" (
  for /f "usebackq delims=" %%P in ("%DISCOVERY_FILE%") do (
    call :try_python "%%P" "generic Windows / Conda discovery"
    if defined PYTHON_CMD goto :DISCOVERY_DONE
  )
)

:DISCOVERY_DONE
if exist "%DISCOVERY_FILE%" del /q "%DISCOVERY_FILE%" >nul 2>&1
exit /b 0
:scan_conda_root
set "CONDA_ROOT_CANDIDATE=%~1"
if "%CONDA_ROOT_CANDIDATE%"=="" exit /b 0
if not exist "%CONDA_ROOT_CANDIDATE%" exit /b 0

call :try_python "%CONDA_ROOT_CANDIDATE%\python.exe" "generic Conda root"
if defined PYTHON_CMD exit /b 0

if exist "%CONDA_ROOT_CANDIDATE%\envs" (
  for /d %%E in ("%CONDA_ROOT_CANDIDATE%\envs\*") do (
    call :try_python "%%~fE\python.exe" "generic Conda environment"
    if defined PYTHON_CMD exit /b 0
  )
)
exit /b 0
