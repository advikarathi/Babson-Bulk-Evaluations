@echo off
REM Batch file to run the grading script automatically
cd /d "%~dp0"
echo Starting Babson College AI Grading Assistant...
python bulk_grade.py
echo.
echo Grading completed! Check the output file.
pause