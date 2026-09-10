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
    try {
        Stop-Process -Id ([int]$lewisPid) -Force -ErrorAction Stop
    }
    catch {
        Write-Host "Could not stop Lewis process ${lewisPid}: $($_.Exception.Message)"
    }

    Start-Sleep -Milliseconds 500

    if (Get-Process -Id ([int]$lewisPid) -ErrorAction SilentlyContinue) {
        # Never delete the PID file while its process is still alive: the
        # tracking must keep pointing at the process that survived.
        Write-Host "Lewis process $lewisPid is still running; keeping the PID file."
        Write-Host "If it was started elevated, run this script from an elevated"
        Write-Host "console or use request-lewis-restart.ps1 instead."
        exit 1
    }

    Write-Host "Lewis process $lewisPid was stopped."
}
else {
    Write-Host "Lewis was not running."
}

Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
