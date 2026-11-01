@echo off
chcp 65001 >nul
title Sub Magic - Launcher

echo ========================================================
echo             SUB MAGIC SYSTEM LAUNCHER
echo ========================================================
echo.

:: 1. Dọn dẹp sạch sẽ các tiến trình cũ trước khi khởi động
echo [INFO] Dọn dẹp tiến trình cũ còn sót lại (Port 8000, 5173)...
call "%~dp0stop.bat" >nul 2>nul

:: 2. Kiểm tra môi trường ảo Python của Backend
if not exist "backend\.venv\Scripts\python.exe" (
    echo [ERROR] Không tìm thấy Python virtual environment tại backend\.venv\
    echo Vui lòng thiết lập môi trường:
    echo   cd backend
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: 3. Kiểm tra Node.js (Windows Native hoặc WSL)
set "USE_WSL=0"
where node >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "USE_WSL=0"
    echo [INFO] Phát hiện Node.js trên môi trường Windows.
) else (
    where wsl >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        wsl node -v >nul 2>nul
        if %ERRORLEVEL% equ 0 (
            set "USE_WSL=1"
            echo [INFO] Phát hiện Node.js trong môi trường WSL.
        )
    )
)

:: 4. Kiểm tra node_modules nếu dùng chế độ Dev
if not exist "frontend\node_modules" (
    echo [NOTICE] Chưa có thư mục frontend\node_modules. Đang cài đặt thư viện...
    if "%USE_WSL%"=="1" (
        wsl --cd "%~dp0frontend" npm install
    ) else (
        cd /d "%~dp0frontend"
        call npm install
        cd /d "%~dp0"
    )
)

:: 5. Khởi động Backend (FastAPI + Uvicorn)
echo [INFO] Khởi động Backend tại http://127.0.0.1:8000 ...
start "Sub Magic Backend" cmd /c "cd /d ""%~dp0backend"" && .venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app"

:: 6. Khởi động Frontend Dev Server
if "%USE_WSL%"=="1" (
    echo [INFO] Khởi động Frontend qua WSL tại http://localhost:5173 ...
    start "Sub Magic Frontend" cmd /c "wsl --cd ""%~dp0frontend"" npm run dev -- --host"
) else (
    echo [INFO] Khởi động Frontend tại http://localhost:5173 ...
    start "Sub Magic Frontend" cmd /c "cd /d ""%~dp0frontend"" && npm run dev"
)

echo [INFO] Đang chờ dịch vụ sẵn sàng...
timeout /t 4 /nobreak >nul

echo [INFO] Đang mở ứng dụng trong trình duyệt...
start http://127.0.0.1:8000

echo.
echo ========================================================
echo  SUB MAGIC ĐANG CHẠY!
echo  - Ứng dụng chính (Nhanh): http://127.0.0.1:8000
echo  - Dev Server:            http://localhost:5173
echo ========================================================
echo.
echo  [LƯU Ý QUAN TRỌNG]:
echo  - Giữ cửa sổ này trong suốt thời gian bạn sử dụng web.
echo  - Khi DÙNG XONG, nhấn PHÍM BẤT KỲ tại cửa sổ này
echo    (hoặc chạy file stop.bat) để TẮT HOÀN TOÀN TẤT CẢ TIẾN TRÌNH.
echo ========================================================
pause >nul

echo [INFO] Đang tắt toàn bộ tiến trình Sub Magic...
call "%~dp0stop.bat"
