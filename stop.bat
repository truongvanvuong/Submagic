@echo off
chcp 65001 >nul
title Sub Magic - Dừng ứng dụng

echo ========================================================
echo        DỪNG TOÀN BỘ TIẾN TRÌNH SUB MAGIC
echo ========================================================
echo [INFO] Đang đóng Backend (Port 8001) và Frontend (Port 3000, 5173)...

:: 1. Kill native Windows node/python processes listening on port 8001, 3000 and 5173 (never kill wslrelay.exe!)
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8001, 3000, 5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { $p = Get-Process -Id $_ -ErrorAction SilentlyContinue; if ($p -and $p.ProcessName -in @('node','python')) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }" >nul 2>nul

:: 2. If WSL is active, kill any Vite dev processes inside WSL
where wsl >nul 2>nul
if %ERRORLEVEL% equ 0 (
    wsl pkill -f "vite|5173" >nul 2>nul
)

:: 3. Close by window title if any remains
taskkill /FI "WINDOWTITLE eq Sub Magic Backend*" /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq Sub Magic Frontend*" /F >nul 2>nul

echo [OK] Đã tắt sạch toàn bộ tiến trình! Không còn tiến trình chạy ngầm.
echo ========================================================
timeout /t 3 /nobreak >nul
