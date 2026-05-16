@echo off
setlocal enabledelayedexpansion

echo Setting up data directory structure...

set FOLDERS=cleaned features\mfcc features\spectrogram features\waveform raw splits

for %%f in (%FOLDERS%) do (
    if not exist "%%f" (
        echo Creating directory: %%f
        mkdir "%%f"
    ) else (
        echo Directory already exists: %%f
    )
)

echo Setup complete!
pause
