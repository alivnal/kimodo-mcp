@echo off
setlocal
title Kimodo
:: Starts the Kimodo demo (http://localhost:7860, or the next free port).
:: Extra arguments go to the demo, e.g.  start_kimodo.bat --offload
set "ROOT=%~dp0"
if not exist "%ROOT%venv311\Scripts\python.exe" (
    echo venv311 not found. Run install_kimodo.bat first.
    pause
    exit /b 1
)
if not exist "%ROOT%kimodo\kimodo-viser\src\viser\client\build\index.html" (
    echo The viser web client is not built. Run install_kimodo.bat first.
    pause
    exit /b 1
)
call "%ROOT%venv311\Scripts\activate.bat"
cd /d "%ROOT%kimodo"
python -m kimodo.demo %*
pause
