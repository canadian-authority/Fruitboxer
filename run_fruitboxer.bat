@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PY="

if exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    set "PY=%PROJECT_DIR%.venv\Scripts\python.exe"
)

if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)

if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo Could not find Python.
    echo Run setup_venv.bat, or install Python 3.10+ and dependencies from requirements.txt.
    exit /b 1
)

pushd "%PROJECT_DIR%" >nul
%PY% -B "%PROJECT_DIR%fruitboxer.py" %*
set "STATUS=%ERRORLEVEL%"
popd >nul
exit /b %STATUS%
