$Repo = "C:\Users\Example\OneDrive\Desktop\Lewis"
$PidFile = "$Repo\runtime\lewis-web.pid"
$StdoutLog = "$Repo\logs\lewis-web.log"
$StderrLog = "$Repo\logs\lewis-web-error.log"

Write-Host "=== Lewis Status ==="

# Lewis web status
try {
    $response = Invoke-WebRequest `
        -Uri "http://127.0.0.1:8787" `
        -UseBasicParsing `
        -TimeoutSec 2 `
        -ErrorAction Stop

    Write-Host "Lewis web: RUNNING (HTTP $($response.StatusCode))"
}
catch {
    Write-Host "Lewis web: NOT RESPONDING"
}

# Lewis process status
if (Test-Path $PidFile) {
    $lewisPid = Get-Content $PidFile -ErrorAction SilentlyContinue

    if (
        $lewisPid -and
        (Get-Process -Id ([int]$lewisPid) -ErrorAction SilentlyContinue)
    ) {
        Write-Host "Lewis PID: $lewisPid (running)"
    }
    else {
        Write-Host "Lewis PID file exists, but the process is not running."
    }
}
else {
    Write-Host "Lewis PID: No PID file found."
}

# LM Studio API and loaded models
try {
    $models = Invoke-RestMethod `
        -Uri "http://127.0.0.1:1234/v1/models" `
        -TimeoutSec 2 `
        -ErrorAction Stop

    Write-Host "LM Studio API: RUNNING"
    Write-Host "Loaded models:"

    foreach ($model in $models.data) {
        Write-Host "  - $($model.id)"
    }
}
catch {
    Write-Host "LM Studio API: NOT RESPONDING"
}

Write-Host ""
Write-Host "Web UI:      http://127.0.0.1:8787"
Write-Host "Standard log: $StdoutLog"
Write-Host "Error log:    $StderrLog"
