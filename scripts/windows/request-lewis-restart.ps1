# Request a Lewis restart from the elevated "Lewis Always On" scheduled task.
#
# A non-elevated session cannot stop an elevated Lewis process directly, so
# this script only drops a marker file. The scheduled task runs
# start-lewis.ps1 with highest privileges every five minutes; when it sees the
# marker it stops the tracked Lewis process, starts a fresh one, and removes
# the marker only after the new process is healthy. This script never stops,
# starts, or kills any process itself.

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$RuntimeDir = "$Repo\runtime"
$Marker = "$RuntimeDir\lewis-restart-requested"

New-Item -ItemType Directory -Force $RuntimeDir | Out-Null
"restart requested at $(Get-Date -Format o)" | Set-Content $Marker -Encoding ascii

Write-Host "Lewis restart requested."
Write-Host "Marker: $Marker"

# Best effort: ask the scheduler to run the task now instead of waiting for
# the next five-minute tick. The task still runs in its own elevated context.
try {
    Start-ScheduledTask -TaskName "Lewis Always On" -ErrorAction Stop
    Write-Host "'Lewis Always On' task triggered; restart should complete shortly."
}
catch {
    Write-Host "Could not trigger the task directly ($($_.Exception.Message))."
    Write-Host "The task's five-minute schedule will pick the marker up."
}
