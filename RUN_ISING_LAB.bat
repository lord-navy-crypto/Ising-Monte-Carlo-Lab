@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  py -3 -m venv .venv
  if errorlevel 1 goto :error
)

echo Installing or checking dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

for /f %%P in ('".venv\Scripts\python.exe" tools\find_free_port.py') do set "PORT=%%P"
if not defined PORT goto :porterror

echo Starting Ising Monte Carlo Lab at http://localhost:%PORT%
".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.port %PORT%
goto :eof

:porterror
echo.
echo No free local port was found in the configured range.
pause
exit /b 1

:error
echo.
echo Setup failed. Review the message above.
pause
exit /b 1
