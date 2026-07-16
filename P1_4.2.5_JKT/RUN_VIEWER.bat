@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo === Proto1 DeckGL Jakarta Focus Pit ===
echo.

where node >nul 2>nul
if errorlevel 1 (
  echo ERROR: Node.js is not installed or is not on PATH.
  echo Install the current Node.js LTS release, reopen this terminal, then run this file again.
  goto :end
)

where npm >nul 2>nul
if errorlevel 1 (
  echo ERROR: npm is not available even though Node.js was found.
  echo Reinstall Node.js LTS, reopen this terminal, then run this file again.
  goto :end
)

if not exist "node_modules\" (
  echo First run in this template: installing locked viewer dependencies...
  call npm install
  if errorlevel 1 (
    echo.
    echo ERROR: npm install failed.
    goto :end
  )
)

echo.
echo Building runtime assets and starting the viewer...
call npm run dev
if errorlevel 1 (
  echo.
  echo ERROR: Viewer launch failed. Read the message above.
)

:end
echo.
pause
