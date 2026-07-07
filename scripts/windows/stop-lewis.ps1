$Repo = "C:\Users\Example\OneDrive\Desktop\Lewis"
$PidFile = "$Repo\runtime\lewis-web.pid"

if (-not (Test-Path $PidFile)) {
    Write-Host "Lewis PID file was not found."
    exit 0
}

$lewisPid = Get-Content $PidFile -ErrorAction SilentlyContinue

if (
    $lewisPid -and
    (Get-Process -Id ([int]$lewisPid) -ErrorAction SilentlyContinue)
) {
    Stop-Process -Id ([int]$lewisPid) -Force
    Write-Host "Lewis process $lewisPid was stopped."
}
else {
    Write-Host "Lewis was not running."
}

Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
