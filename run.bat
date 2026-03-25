@echo off
set SCRIPT_LOC=%~dp0
call "%SCRIPT_LOC%.venv\Scripts\activate.bat"
python "%SCRIPT_LOC%run.py" %*
REM: If there is any error, pause the window to allow reading what went wrong
if %errorlevel% neq 0 pause