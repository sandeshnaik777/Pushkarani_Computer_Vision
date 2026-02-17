$env:NODE_OPTIONS = "--openssl-legacy-provider"
$env:PORT = 3000
$env:BROWSER = "none"
$env:SKIP_PREFLIGHT_CHECK = "true"
$env:REACT_APP_API_URL = "http://localhost:5000/api"

Write-Host "Starting React frontend with NODE_OPTIONS: $($env:NODE_OPTIONS)" -ForegroundColor Green
Write-Host "Port: $($env:PORT)" -ForegroundColor Green
Write-Host "API URL: $($env:REACT_APP_API_URL)" -ForegroundColor Green

npm start
