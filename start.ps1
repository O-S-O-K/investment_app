# start.ps1 - one-click launcher for Investment App
# Run from the investment_app folder: .\start.ps1

$root = $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$envFile = Join-Path $root ".env"
$envExample = Join-Path $root ".env.example"

# Create .env if missing
if (-not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Host ".env created from .env.example - edit API_KEY before exposing on a network" -ForegroundColor Yellow
}

# Create venv and install if missing
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv "$root\.venv"
    & $venvPython -m pip install --quiet --upgrade pip
    & $venvPython -m pip install --quiet -e $root
}

# Get LAN IP for phone access
$lanIP = (
    Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.*" } |
    Select-Object -First 1
).IPAddress

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "  Investment App - Local Launcher" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "  API       -> http://127.0.0.1:8000"
Write-Host "  Dashboard -> http://127.0.0.1:8501"
if ($lanIP) {
    Write-Host "  Phone (LAN) -> http://${lanIP}:8501" -ForegroundColor Cyan
}
Write-Host ""

# Start API in a new window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root'; Write-Host 'API Server' -ForegroundColor Green; & '$venvPython' -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
)

# Give API a moment to bind
Start-Sleep -Seconds 3

# Start Streamlit in a new window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$root'; Write-Host 'Dashboard' -ForegroundColor Green; & '$venvPython' -m streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501"
)

Write-Host "Both servers starting in separate windows." -ForegroundColor Green
Write-Host "Set API Base URL in Streamlit sidebar to: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "Press any key to exit this launcher..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
