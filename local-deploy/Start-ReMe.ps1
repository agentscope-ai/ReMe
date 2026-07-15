[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$ReMeRoot = 'D:\projects\reme'
$DataRoot = 'D:\projects\reme-data'
$Executable = Join-Path $ReMeRoot '.venv\Scripts\reme.exe'
$LogRoot = Join-Path $ReMeRoot 'local-deploy\logs'
$PidFile = Join-Path $LogRoot 'reme.pid'
$Port = 2333

if (-not (Test-Path -LiteralPath $Executable)) {
    throw "ReMe executable not found: $Executable"
}

$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    $owners = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $owner"
        if ($process.CommandLine -like "*$ReMeRoot*" -and $process.CommandLine -match '\bstart\b') {
            Write-Output "ReMe is already running on 127.0.0.1:$Port (PID $owner)."
            exit 0
        }
    }
    throw "Port $Port is already owned by another process."
}

New-Item -ItemType Directory -Force -Path $DataRoot, $LogRoot | Out-Null

# Keep optional model-backed jobs unavailable in this deployment process.
foreach ($name in @('LLM_API_KEY', 'EMBEDDING_API_KEY', 'CLAUDE_CODE_API_KEY')) {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$env:NO_PROXY = '127.0.0.1,localhost,::1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$arguments = @(
    'start'
    "workspace_dir=$DataRoot"
    'service.host=127.0.0.1'
    "service.port=$Port"
)

$process = Start-Process `
    -FilePath $Executable `
    -ArgumentList $arguments `
    -WorkingDirectory $ReMeRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogRoot 'reme.out.log') `
    -RedirectStandardError (Join-Path $LogRoot 'reme.err.log') `
    -PassThru

Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
Start-Sleep -Milliseconds 750
if ($process.HasExited) {
    $errorLog = Join-Path $LogRoot 'reme.err.log'
    $details = if (Test-Path -LiteralPath $errorLog) {
        (Get-Content -LiteralPath $errorLog -Tail 20 -Encoding UTF8) -join [Environment]::NewLine
    } else {
        'No error log was produced.'
    }
    throw "ReMe exited during startup.$([Environment]::NewLine)$details"
}

Write-Output "Started ReMe on 127.0.0.1:$Port (PID $($process.Id))."
