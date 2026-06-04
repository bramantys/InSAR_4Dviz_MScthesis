@echo off
setlocal EnableExtensions

rem ============================================================================
rem run_baby_run.bat
rem Portable InSAR4D RUM Viewer Pipeline launcher
rem
rem Expected location:
rem   <project_root>\config\run_baby_run.bat
rem
rem Python selection priority:
rem   1. INSAR4D_PYTHON, if user defined it
rem   2. active conda environment via CONDA_PREFIX
rem   3. project-local .venv
rem   4. project-local venv
rem   5. user-local Miniconda / Anaconda / Miniforge / Mambaforge
rem   6. conda base via `conda info --base`, if conda is on PATH
rem   7. python from active terminal / PATH
rem
rem This file intentionally does NOT hard-code any personal Windows username.
rem ============================================================================

set "CONFIG_DIR=%~dp0"
for %%I in ("%CONFIG_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "RUNNER=%PROJECT_ROOT%\_internal\pipeline\00_run_pipeline.py"
set "REQUIRED_PACKAGES=pandas pyproj pillow"
set "PYTHONUTF8=1"

call :select_python

echo.
echo ================================================================================
echo  InSAR4D RUM Viewer Pipeline - run_baby_run
echo ================================================================================
echo  Project root  : %PROJECT_ROOT%
echo  Runner        : %RUNNER%
if "%PYTHON_CMD%"=="" (
echo  Python        : NOT FOUND
) else (
echo  Python        : %PYTHON_CMD%
echo  Python source : %PYTHON_SOURCE%
)
echo ================================================================================

if not exist "%RUNNER%" (
    echo.
    echo [ERROR] Pipeline runner not found:
    echo         %RUNNER%
    echo.
    echo Make sure this launcher is located inside:
    echo         ^<project_root^>\config\run_baby_run.bat
    echo.
    pause
    exit /b 1
)

if "%PYTHON_CMD%"=="" (
    call :python_not_found
    exit /b 1
)

call :check_python_packages
if errorlevel 1 (
    call :missing_packages_flow %*
    if errorlevel 1 exit /b 1
)

echo.
echo [CHECK] Checking project config and data...

if not exist "%PROJECT_ROOT%\config\project_config.json" (
    echo.
    echo [ERROR] project_config.json not found.
    echo         Expected: %PROJECT_ROOT%\config\project_config.json
    echo.
    pause
    exit /b 1
)

"%PYTHON_CMD%" -c "import json,sys; d=json.load(open(r'%PROJECT_ROOT%\config\project_config.json')); print(d['user_inputs']['source_file'])" > "%TEMP%\_insar4d_src.txt" 2>nul
set /p SOURCE_FILE=<"%TEMP%\_insar4d_src.txt"
del "%TEMP%\_insar4d_src.txt" 2>nul

if "%SOURCE_FILE%"=="" (
    echo.
    echo [ERROR] Could not read user_inputs.source_file from project_config.json.
    echo         Check that the field exists and the JSON is valid.
    echo.
    pause
    exit /b 1
)

if not exist "%PROJECT_ROOT%\%SOURCE_FILE%" (
    echo.
    echo [ERROR] Data file not found: %SOURCE_FILE%
    echo         Config points to: %PROJECT_ROOT%\%SOURCE_FILE%
    echo         Either place your data file there, or fix source_file in config\project_config.json
    echo.
    pause
    exit /b 1
)

echo [CHECK] OK - config found, data file found: %SOURCE_FILE%

rem If command-line arguments were provided, pass them directly to 00_run_pipeline.py.
if not "%~1"=="" (
    echo.
    echo [MODE] Direct arguments detected:
    echo        %*
    echo.
    pushd "%PROJECT_ROOT%"
    "%PYTHON_CMD%" "%RUNNER%" %*
    set "EXIT_CODE=%ERRORLEVEL%"
    popd
    goto :finish
)

:menu
echo.
echo Choose what to run:
echo.
echo   [1] Full pipeline 01-18
echo   [2] Dry run / planned steps only
echo   [3] List steps
echo   [4] Run one step
echo   [5] Run step range
echo   [6] Rerun final viewer tuning only ^(Step 18^)
echo   [S] Setup/update project .venv
echo   [Q] Quit
echo.
set /p "CHOICE=Your choice: "

if /I "%CHOICE%"=="1" goto :full
if /I "%CHOICE%"=="2" goto :dry
if /I "%CHOICE%"=="3" goto :list
if /I "%CHOICE%"=="4" goto :one
if /I "%CHOICE%"=="5" goto :range
if /I "%CHOICE%"=="6" goto :step18
if /I "%CHOICE%"=="S" goto :setup
if /I "%CHOICE%"=="Q" goto :quit

echo.
echo [WARN] Unknown choice: %CHOICE%
goto :menu

:full
set "RUN_ARGS="
goto :run

:dry
set "RUN_ARGS=--dry-run"
goto :run

:list
set "RUN_ARGS=--list-steps"
goto :run

:one
echo.
set /p "STEP_ONE=Step number to run ^(example: 1, 03, or 18^): "
if "%STEP_ONE%"=="" (
    echo [WARN] No step entered.
    goto :menu
)
call :normalize_step "%STEP_ONE%" STEP_ONE_NORM
set "RUN_ARGS=--from-step %STEP_ONE_NORM% --to-step %STEP_ONE_NORM%"
goto :run

:range
echo.
set /p "STEP_FROM=From step ^(example: 1 or 01^): "
set /p "STEP_TO=To step   ^(example: 18^): "
if "%STEP_FROM%"=="" (
    echo [WARN] No from-step entered.
    goto :menu
)
if "%STEP_TO%"=="" (
    echo [WARN] No to-step entered.
    goto :menu
)
call :normalize_step "%STEP_FROM%" STEP_FROM_NORM
call :normalize_step "%STEP_TO%" STEP_TO_NORM
set "RUN_ARGS=--from-step %STEP_FROM_NORM% --to-step %STEP_TO_NORM%"
goto :run

:step18
set "RUN_ARGS=--from-step 18 --to-step 18"
goto :run

:setup
call :setup_venv
if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed.
    goto :menu
)

call :select_python
call :check_python_packages
if errorlevel 1 (
    echo.
    echo [ERROR] Setup finished, but package check still failed.
    goto :menu
)

echo.
echo [OK] Project .venv is ready.
goto :menu

:run
call :check_python_packages
if errorlevel 1 (
    call :missing_packages_flow
    if errorlevel 1 goto :menu
)

echo.
echo ================================================================================
echo  Running:
echo  "%PYTHON_CMD%" "%RUNNER%" %RUN_ARGS%
echo ================================================================================
echo.
pushd "%PROJECT_ROOT%"
"%PYTHON_CMD%" "%RUNNER%" %RUN_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd
goto :finish

:finish
echo.
echo ================================================================================
if "%EXIT_CODE%"=="0" (
    echo  run_baby_run finished.
) else (
    echo  run_baby_run stopped with error code %EXIT_CODE%.
)
echo ================================================================================
echo.
echo Useful files:
echo   run_records\latest_run.log
echo   run_records\latest_manifest.json
echo   _internal\data_pipeline\viewer_resources\viewer_tuning.json
echo.

rem If called with direct command-line arguments, return to terminal normally.
if not "%~1"=="" exit /b %EXIT_CODE%

:after_run_menu
echo What next?
echo.
echo   [M] Back to start menu
echo   [Q] Quit
echo.
set /p "AFTER_CHOICE=Your choice: "

if /I "%AFTER_CHOICE%"=="M" goto :menu
if /I "%AFTER_CHOICE%"=="Q" goto :quit

echo.
echo [WARN] Unknown choice: %AFTER_CHOICE%
goto :after_run_menu

:quit
echo.
echo Bye.
echo.
exit /b 0

rem ============================================================================
rem Subroutines below this line.
rem Main flow exits before this section, so labels are only reached via CALL.
rem ============================================================================

:select_python
set "PYTHON_CMD="
set "PYTHON_SOURCE="

rem 1. Explicit user override.
if not "%INSAR4D_PYTHON%"=="" (
    set "PYTHON_CMD=%INSAR4D_PYTHON%"
    set "PYTHON_SOURCE=INSAR4D_PYTHON"
    exit /b 0
)

rem 2. If launched from an activated conda environment, use that exact env.
if not "%CONDA_PREFIX%"=="" (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON_CMD=%CONDA_PREFIX%\python.exe"
        set "PYTHON_SOURCE=active conda env"
        exit /b 0
    )
)

rem 3. Project-local virtual environments should win over global conda base.
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_ROOT%\.venv\Scripts\python.exe"
    set "PYTHON_SOURCE=project .venv"
    exit /b 0
)

rem 4. Legacy project venv folder.
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_ROOT%\venv\Scripts\python.exe"
    set "PYTHON_SOURCE=project venv"
    exit /b 0
)

rem 5. Common user-local Conda-family installs. Uses USERPROFILE/LOCALAPPDATA, not a hard-coded username.
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PYTHON_CMD=%USERPROFILE%\miniconda3\python.exe"
    set "PYTHON_SOURCE=user Miniconda base"
    exit /b 0
)

if exist "%LOCALAPPDATA%\miniconda3\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\miniconda3\python.exe"
    set "PYTHON_SOURCE=localappdata Miniconda base"
    exit /b 0
)

if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PYTHON_CMD=%USERPROFILE%\anaconda3\python.exe"
    set "PYTHON_SOURCE=user Anaconda base"
    exit /b 0
)

if exist "%USERPROFILE%\miniforge3\python.exe" (
    set "PYTHON_CMD=%USERPROFILE%\miniforge3\python.exe"
    set "PYTHON_SOURCE=user Miniforge base"
    exit /b 0
)

if exist "%USERPROFILE%\mambaforge\python.exe" (
    set "PYTHON_CMD=%USERPROFILE%\mambaforge\python.exe"
    set "PYTHON_SOURCE=user Mambaforge base"
    exit /b 0
)

rem 6. If conda is on PATH, ask conda where base lives.
conda --version >nul 2>&1
if not errorlevel 1 (
    for /f "usebackq delims=" %%C in (`conda info --base 2^>nul`) do (
        if exist "%%C\python.exe" (
            set "PYTHON_CMD=%%C\python.exe"
            set "PYTHON_SOURCE=conda base from conda info"
            exit /b 0
        )
    )
)

rem 7. Last fallback: whatever python is active in this terminal/PATH.
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    set "PYTHON_SOURCE=PATH / active terminal"
    exit /b 0
)

exit /b 0

:check_python_packages
if "%PYTHON_CMD%"=="" exit /b 1

"%PYTHON_CMD%" --version >nul 2>&1
if errorlevel 1 exit /b 1

"%PYTHON_CMD%" -c "import pandas, pyproj, PIL" >nul 2>&1
if errorlevel 1 exit /b 1

exit /b 0

:missing_packages_flow
echo.
echo [ERROR] This Python can start, but required packages are missing.
echo.
echo Selected Python:
echo   %PYTHON_CMD%
echo.
echo Required packages:
echo   %REQUIRED_PACKAGES%
echo.

rem Direct argument mode should not trap user in an interactive setup menu.
if not "%~1"=="" (
    echo Install with:
    echo   "%PYTHON_CMD%" -m pip install %REQUIRED_PACKAGES%
    echo.
    echo Or create a project-local environment:
    echo   cd "%PROJECT_ROOT%"
    echo   "%PYTHON_CMD%" -m venv .venv
    echo   .venv\Scripts\python -m pip install %REQUIRED_PACKAGES%
    echo.
    exit /b 1
)

:missing_pkg_menu
echo What do you want to do?
echo.
echo   [I] Install required packages into selected Python
echo   [S] Create/update project .venv automatically
echo   [P] Use a custom python.exe path for this session
echo   [Q] Quit
echo.
set /p "PKG_CHOICE=Your choice: "

if /I "%PKG_CHOICE%"=="I" (
    call :install_into_selected_python
    if errorlevel 1 exit /b 1
    call :check_python_packages
    exit /b %ERRORLEVEL%
)

if /I "%PKG_CHOICE%"=="S" (
    call :setup_venv
    if errorlevel 1 exit /b 1
    call :select_python
    call :check_python_packages
    exit /b %ERRORLEVEL%
)

if /I "%PKG_CHOICE%"=="P" (
    call :custom_python
    call :check_python_packages
    exit /b %ERRORLEVEL%
)

if /I "%PKG_CHOICE%"=="Q" exit /b 1

echo.
echo [WARN] Unknown choice: %PKG_CHOICE%
goto :missing_pkg_menu

:python_not_found
echo.
echo [ERROR] Python was not found.
echo.
echo Install Python first, then run this again.
echo Recommended setup:
echo.
echo   cd "%PROJECT_ROOT%"
echo   python -m venv .venv
echo   .venv\Scripts\python -m pip install %REQUIRED_PACKAGES%
echo   config\run_baby_run.bat
echo.
pause
exit /b 1

:install_into_selected_python
echo.
echo ================================================================================
echo  Installing required packages into selected Python
echo ================================================================================
echo  Python:
echo    %PYTHON_CMD%
echo.
echo  Packages:
echo    %REQUIRED_PACKAGES%
echo ================================================================================
echo.
"%PYTHON_CMD%" -m pip install %REQUIRED_PACKAGES%
if errorlevel 1 (
    echo.
    echo [ERROR] Package installation failed for selected Python.
    exit /b 1
)
exit /b 0

:setup_venv
echo.
echo ================================================================================
echo  Creating/updating project-local Python environment
echo ================================================================================
echo  Location:
echo    %PROJECT_ROOT%\.venv
echo.
echo  Packages:
echo    %REQUIRED_PACKAGES%
echo ================================================================================

pushd "%PROJECT_ROOT%"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [SETUP] Creating .venv using selected Python ...
    "%PYTHON_CMD%" -m venv .venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not create .venv using:
        echo   %PYTHON_CMD%
        popd
        exit /b 1
    )
) else (
    echo.
    echo [SETUP] Existing .venv found.
)

echo.
echo [SETUP] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo [WARN] pip upgrade failed. Continuing to package install anyway.
)

echo.
echo [SETUP] Installing required packages ...
".venv\Scripts\python.exe" -m pip install %REQUIRED_PACKAGES%
if errorlevel 1 (
    echo.
    echo [ERROR] Package installation failed.
    echo Check internet connection or install packages manually:
    echo   ".venv\Scripts\python.exe" -m pip install %REQUIRED_PACKAGES%
    popd
    exit /b 1
)

popd

set "PYTHON_CMD=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "PYTHON_SOURCE=project .venv"
exit /b 0

:custom_python
echo.
set /p "CUSTOM_PY=Paste full path to python.exe: "
if "%CUSTOM_PY%"=="" (
    echo [WARN] No path entered.
    exit /b 1
)

if not exist "%CUSTOM_PY%" (
    echo [ERROR] File does not exist:
    echo   %CUSTOM_PY%
    exit /b 1
)

set "PYTHON_CMD=%CUSTOM_PY%"
set "PYTHON_SOURCE=custom session path"
exit /b 0

:normalize_step
rem Normalizes step input:
rem   1  -> 01
rem   01 -> 01
rem   18 -> 18
setlocal
set "RAW_STEP=%~1"
set "RAW_STEP=%RAW_STEP: =%"
if "%RAW_STEP%"=="1" set "RAW_STEP=01"
if "%RAW_STEP%"=="2" set "RAW_STEP=02"
if "%RAW_STEP%"=="3" set "RAW_STEP=03"
if "%RAW_STEP%"=="4" set "RAW_STEP=04"
if "%RAW_STEP%"=="5" set "RAW_STEP=05"
if "%RAW_STEP%"=="6" set "RAW_STEP=06"
if "%RAW_STEP%"=="7" set "RAW_STEP=07"
if "%RAW_STEP%"=="8" set "RAW_STEP=08"
if "%RAW_STEP%"=="9" set "RAW_STEP=09"
endlocal & set "%~2=%RAW_STEP%"
exit /b 0
