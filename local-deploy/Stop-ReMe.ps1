[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$ReMeRoot = 'D:\projects\reme'
$LogRoot = Join-Path $ReMeRoot 'local-deploy\logs'
$PidFile = Join-Path $LogRoot 'reme.pid'
$Port = 2333
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)

if ($listeners.Count -eq 0) {
    Write-Output "No listener is running on port $Port."
    exit 0
}

$owners = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($owner in $owners) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $owner"
    if ($null -eq $process) {
        continue
    }
    if ($process.CommandLine -notlike "*$ReMeRoot*" -or $process.CommandLine -notmatch '\bstart\b') {
        throw "Refusing to stop PID $owner because it is not the local ReMe service."
    }
    Stop-Process -Id $owner -Force
    Write-Output "Stopped ReMe process $owner."
}

if (Test-Path -LiteralPath $PidFile) {
    Remove-Item -LiteralPath $PidFile -Force
}
