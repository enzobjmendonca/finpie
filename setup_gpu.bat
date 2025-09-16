@echo off
echo Installing FinPie with CUDA GPU Support...
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo Step 4: Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo Error: Failed to install PyTorch with CUDA
    pause
    exit /b 1
)

echo Step 5: Installing remaining requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo Step 6: Installing Jupyter kernel...
python -m ipykernel install --user --name=finpie --display-name="Python (finpie)"

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To verify GPU is working, run:
echo   venv\Scripts\activate.bat
echo   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo.
echo To start Jupyter:
echo   venv\Scripts\activate.bat
echo   jupyter notebook
echo.
pause
