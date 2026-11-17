@echo off
rem Builds dist\SudokuReader\SudokuReader.exe (a double-clickable app, no Python needed to run it).
cd /d "%~dp0"
set PYTHONPATH=%PYTHONPATH%;%~dp0
python prepare_tcl.py
set TCL=
if exist build_tcl\tcl_library set TCL=--add-data "build_tcl\tcl_library;_tcl_data" --add-data "build_tcl\tk_library;_tk_data"
python -m PyInstaller --noconfirm --windowed --name SudokuReader --collect-all customtkinter %TCL% ^
  --add-data "nn\model_mlp.npz;nn" --add-data "nn\model_logreg.npz;nn" --add-data "data\mnist19.npz;data" ^
  --add-data "eval\dashboard_template.html;eval" --add-data "outputs;outputs" app.py
echo.
echo Done: dist\SudokuReader\SudokuReader.exe
pause
