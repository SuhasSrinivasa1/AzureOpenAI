@echo off
REM === Activate Anaconda base ===
CALL "C:\Program Files\Anaconda3\Scripts\activate.bat" base

REM === Go to your app folder ===
cd /d "C:\Users\IJJ3KOR\LLMHelper\AzureOpenAI"

REM === Launch Streamlit using the interpreter directly ===
"C:\Program Files\Anaconda3\python.exe" -m streamlit run streamlit_app.py

pause
