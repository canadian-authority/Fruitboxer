@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "BASE_PY="

where py >nul 2>nul
if not errorlevel 1 (
    set "BASE_PY=py -3"
)

if not defined BASE_PY (
    where python >nul 2>nul
    if not errorlevel 1 set "BASE_PY=python"
)

if not defined BASE_PY (
    echo Could not find a base Python interpreter.
    echo Install Python 3.10+ and make the py launcher or python command available.
    exit /b 1
)

%BASE_PY% -m venv "%PROJECT_DIR%.venv"
if errorlevel 1 exit /b %ERRORLEVEL%

"%PROJECT_DIR%.venv\Scripts\python.exe" -m pip install -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 exit /b %ERRORLEVEL%

echo.
echo Virtual environment ready. Run:
echo   run_fruitboxer.bat --no-execute --print-moves
