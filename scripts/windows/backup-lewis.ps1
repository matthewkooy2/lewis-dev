$Repo = "C:\Users\Example\OneDrive\Desktop\Lewis"
$Python = "$Repo\.venv\Scripts\python.exe"
$BackupScript = "$Repo\scripts\windows\backup-lewis.py"
$Log = "$Repo\logs\backup-lewis.log"

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting backup." |
    Add-Content $Log

& $Python $BackupScript 2>&1 |
    Tee-Object -FilePath $Log -Append

if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Backup FAILED." |
        Add-Content $Log
    exit $LASTEXITCODE
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Backup completed." |
    Add-Content $Log
