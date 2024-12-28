@echo off
chcp 65001

set USE_MIRROR=true
set PYTHONPATH=%~dp0
set PYTHON_CMD=python
if exist "anyaenv" (
    set PYTHON_CMD=%cd%\anyaenv\env\python
)

setlocal enabledelayedexpansion

set "HF_ENDPOINT=https://huggingface.co"
set "no_proxy="
if "%USE_MIRROR%" == "true" (
    set "HF_ENDPOINT=https://hf-mirror.com"
    set "no_proxy=localhost, 127.0.0.1, 0.0.0.0"
)
echo "HF_ENDPOINT: !HF_ENDPOINT!"
echo "NO_PROXY: !no_proxy!"

echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~\u4E00-\u9FFF ] " >nul && (
    echo.
    echo There are special characters in the current path, please make the path of fish-speech free of special characters before running. && (
        goto end
    )
)

echo.
%PYTHON_CMD% main.py


:end
endlocal
pause
