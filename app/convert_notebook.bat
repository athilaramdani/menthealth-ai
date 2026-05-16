@echo off
setlocal

echo =======================================================
echo KONVERTER PYTHON (.py) KE JUPYTER NOTEBOOK (.ipynb)
echo =======================================================
echo.

:: Langsung oper semua argumen ke script Python.
:: Jika kosong, script Python otomatis membuka File Explorer (GUI).
python py_to_ipynb.py %1 %2

echo.
pause
