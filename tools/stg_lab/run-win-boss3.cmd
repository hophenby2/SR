@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-win-boss3.ps1" %*
set "STG_LAB_EXIT=%ERRORLEVEL%"

echo.
if "%STG_LAB_EXIT%"=="0" (
    echo Boss 3 test completed successfully.
) else (
    echo Boss 3 test failed with exit code %STG_LAB_EXIT%.
)
echo.
pause
exit /b %STG_LAB_EXIT%
