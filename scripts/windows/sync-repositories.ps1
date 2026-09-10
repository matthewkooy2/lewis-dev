$ErrorActionPreference = "Stop"

$LewisRepo = "C:\Users\Example\OneDrive\Desktop\Lewis"
$ProjectsRoot = "C:\Projects"
$LogFile = "$LewisRepo\logs\repo-sync.log"

function Write-SyncLog {
    param([string]$Message)

    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    $line | Add-Content $LogFile
    Write-Host $line
}

function Sync-Repository {
    param([string]$Repo)

    if (-not (Test-Path (Join-Path $Repo ".git"))) {
        return $false
    }

    $name = Split-Path $Repo -Leaf

    $dirty = (& git -C $Repo status --porcelain | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        Write-SyncLog "${name}: unable to read Git status."
        return $false
    }

    if ($dirty) {
        Write-SyncLog "${name}: skipped because the working tree has local changes."
        return $false
    }

    & git -C $Repo fetch origin --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-SyncLog "${name}: fetch failed."
        return $false
    }

    $branch = (& git -C $Repo branch --show-current | Out-String).Trim()
    if (-not $branch) {
        Write-SyncLog "${name}: skipped because it is not on a branch."
        return $false
    }

    & git -C $Repo show-ref --verify --quiet "refs/remotes/origin/$branch"
    if ($LASTEXITCODE -ne 0) {
        Write-SyncLog "${name}: origin/$branch does not exist."
        return $false
    }

    $behind = [int](
        (& git -C $Repo rev-list --count "HEAD..origin/$branch" | Out-String).Trim()
    )

    $ahead = [int](
        (& git -C $Repo rev-list --count "origin/$branch..HEAD" | Out-String).Trim()
    )

    if ($behind -eq 0) {
        Write-SyncLog "${name}: already up to date."
        return $false
    }

    if ($ahead -gt 0) {
        Write-SyncLog "${name}: skipped because local and remote history differ."
        return $false
    }

    Write-SyncLog "${name}: pulling $behind new commit(s)."

    & git -C $Repo pull --ff-only origin $branch |
        ForEach-Object {
            Write-SyncLog "${name} Git: $_"
        }

    if ($LASTEXITCODE -ne 0) {
        Write-SyncLog "${name}: pull failed."
        return $false
    }

    Write-SyncLog "${name}: updated successfully."
    return $true
}

New-Item -ItemType Directory -Force (Split-Path $LogFile) | Out-Null

$lewisUpdated = Sync-Repository $LewisRepo

if (Test-Path $ProjectsRoot) {
    $projectRepos = Get-ChildItem $ProjectsRoot -Directory |
        Where-Object {
            Test-Path (Join-Path $_.FullName ".git")
        }

    foreach ($project in $projectRepos) {
        Sync-Repository $project.FullName | Out-Null
    }
}

if ($lewisUpdated -eq $true) {
    Write-SyncLog "Restarting Lewis after update."

    powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File "$LewisRepo\scripts\windows\stop-lewis.ps1"

    powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File "$LewisRepo\scripts\windows\start-lewis.ps1"
}
