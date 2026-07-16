@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo === Build Proto1 DeckGL release ===
echo.

where node >nul 2>nul
if errorlevel 1 (
  echo ERROR: Node.js is not installed or is not on PATH.
  goto :end
)

if not exist "node_modules\" (
  echo First run in this template: installing locked viewer dependencies...
  call npm install
  if errorlevel 1 goto :end
)

call npm run build
if errorlevel 1 (
  echo.
  echo ERROR: Release build failed. Read the message above.
) else (
  echo.
  echo Release written to dist\
)

:end
echo.
pause
