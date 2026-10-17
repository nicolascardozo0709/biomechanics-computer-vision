@echo off
echo ======================================================================
echo  Kinetix AI -- Launching Interactive SportsTech Web Dashboard
echo  Developer: Nicolas Cardozo (Pontificia Universidad Javeriana)
echo ======================================================================
echo.
echo Starting FastAPI Web Server at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
