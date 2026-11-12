@echo off
cd /d "%~dp0"
echo Starting Music Score Creator Web UI...
echo Open your browser to: http://127.0.0.1:8000
echo Press Ctrl+C in this window when you want to stop the server.
echo.
.\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
pause
