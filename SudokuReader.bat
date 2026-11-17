@echo off
rem Double-click to open the Sudoku Reader app (no console window).
cd /d "%~dp0"
where pythonw >nul 2>nul && (start "" pythonw app.py) || (start "" python app.py)
