$ErrorActionPreference = "Stop"

$Repo = "C:\Users\Example\OneDrive\Desktop\Lewis"
$Python = "$Repo\.venv\Scripts\python.exe"
$Lms = "C:\Users\Example\.lmstudio\bin\lms.exe"
$Model = "qwen2.5-coder-7b-instruct"

$RuntimeDir = "$Repo\runtime"
$LogDir = "$Repo\logs"
$PidFile = "$RuntimeDir\lewis-web.pid"
$StdoutLog = "$LogDir\lewis-web.log"
$StderrLog = "$LogDir\lewis-web-error.log"

New-Item -ItemType Directory -Force $RuntimeDir, $LogDir | Out-Null

function Test-LmStudioApi {
    try {
        Invoke-RestMethod `
            -Uri "http://127.0.0.1:1234/v1/models" `
            -TimeoutSec 2 `
            -ErrorAction Stop | Out-Null

        return $true
    }
    catch {
        return $false
    }
}

function Test-LewisWeb {
    try {
        Invoke-WebRequest `
            -Uri "http://127.0.0.1:8787" `
            -UseBasicParsing `
            -TimeoutSec 2 `
            -ErrorAction Stop | Out-Null

        return $true
    }
    catch {
        return $false
    }
}

# Start LM Studio without blocking forever.
if (-not (Test-LmStudioApi)) {
    Write-Host "Starting LM Studio daemon..."

    Start-Process `
        -FilePath $Lms `
        -ArgumentList @("daemon", "up") `
        -WindowStyle Hidden | Out-Null

    # Give the daemon time to initialize.
    Start-Sleep -Seconds 5

    if (-not (Test-LmStudioApi)) {
        Write-Host "Starting LM Studio API server..."

        Start-Process `
            -FilePath $Lms `
            -ArgumentList @(
                "server",
                "start",
                "--port",
                "1234",
                "--bind",
                "127.0.0.1"
            ) `
            -WindowStyle Hidden | Out-Null
    }
}

# Wait up to 60 seconds for the API.
$apiReady = $false

for ($i = 0; $i -lt 30; $i++) {
    if (Test-LmStudioApi) {
        $apiReady = $true
        break
    }

    # Retry the server command once in case the daemon was not ready initially.
    if ($i -eq 10) {
        Start-Process `
            -FilePath $Lms `
            -ArgumentList @(
                "server",
                "start",
                "--port",
                "1234",
                "--bind",
                "127.0.0.1"
            ) `
            -WindowStyle Hidden | Out-Null
    }

    Start-Sleep -Seconds 2
}

if (-not $apiReady) {
    throw "LM Studio API did not become available on port 1234 within 60 seconds."
}

# Ensure Qwen is loaded.
$models = Invoke-RestMethod `
    -Uri "http://127.0.0.1:1234/v1/models" `
    -TimeoutSec 5

$loadedModelIds = @(
    $models.data | ForEach-Object { $_.id }
)

if ($loadedModelIds -notcontains $Model) {
    Write-Host "Loading $Model..."

    $loadProcess = Start-Process `
        -FilePath $Lms `
        -ArgumentList @(
            "load",
            $Model,
            "--yes"
        ) `
        -WindowStyle Hidden `
        -PassThru

    if (-not $loadProcess.WaitForExit(120000)) {
        Stop-Process `
            -Id $loadProcess.Id `
            -Force `
            -ErrorAction SilentlyContinue

        throw "Loading $Model exceeded the 120-second timeout."
    }

    if ($loadProcess.ExitCode -ne 0) {
        throw "LM Studio failed to load $Model."
    }
}

# Do not start a duplicate Lewis instance.
if (Test-LewisWeb) {
    Write-Host "Lewis is already responding on port 8787."
    exit 0
}

# Remove a stale PID file.
if (Test-Path $PidFile) {
    $existingPid = Get-Content $PidFile -ErrorAction SilentlyContinue

    if (
        $existingPid -and
        (Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue)
    ) {
        Write-Host "Lewis is already running with PID $existingPid."
        exit 0
    }

    Remove-Item $PidFile -Force
}

Write-Host "Starting Lewis..."

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @("-u", "-m", "app.main", "web") `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content $PidFile

# Wait up to 30 seconds for Lewis.
$lewisReady = $false

for ($i = 0; $i -lt 30; $i++) {
    if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
        break
    }

    if (Test-LewisWeb) {
        $lewisReady = $true
        break
    }

    Start-Sleep -Seconds 1
}

if (-not $lewisReady) {
    Stop-Process `
        -Id $process.Id `
        -Force `
        -ErrorAction SilentlyContinue

    Remove-Item `
        $PidFile `
        -Force `
        -ErrorAction SilentlyContinue

    throw "Lewis failed to start. Check: $StderrLog"
}

Write-Host ""
Write-Host "Lewis is running."
Write-Host "Web UI:    http://127.0.0.1:8787"
Write-Host "Model API: http://127.0.0.1:1234"
Write-Host "Lewis PID: $($process.Id)"
Write-Host "Logs:      $LogDir"
