[CmdletBinding()]
param(
    [ValidateRange(1, 600)]
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'

$ReMeRoot = 'D:\projects\reme'
$Executable = Join-Path $ReMeRoot '.venv\Scripts\reme.exe'
$Port = 2333
$Uri = "http://127.0.0.1:$Port/version"
$env:NO_PROXY = '127.0.0.1,localhost,::1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
$version = $null

do {
    try {
        $version = Invoke-RestMethod -Method Post -Uri $Uri -ContentType 'application/json' -Body '{}'
        break
    } catch {
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "ReMe did not become ready within $TimeoutSeconds seconds: $($_.Exception.Message)"
        }
        Start-Sleep -Milliseconds 500
    }
} while ($true)

$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
$invalid = @($listeners | Where-Object { $_.LocalAddress -ne '127.0.0.1' })
if ($invalid.Count -gt 0) {
    $addresses = ($invalid | Select-Object -ExpandProperty LocalAddress -Unique) -join ', '
    throw "ReMe is listening outside loopback: $addresses"
}

$owners = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($owner in $owners) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $owner"
    if ($null -eq $process -or $process.CommandLine -notlike "*$ReMeRoot*") {
        throw "Port $Port is not owned by the expected ReMe deployment."
    }
}

foreach ($action in @('shell', 'auto_memory', 'auto_resource', 'auto_dream')) {
    $blocked = Invoke-WebRequest `
        -SkipHttpErrorCheck `
        -Method Post `
        -Uri "http://127.0.0.1:$Port/$action" `
        -ContentType 'application/json' `
        -Body '{}'
    if ($blocked.StatusCode -ne 404) {
        throw "Unsafe ReMe action remains exposed: $action returned HTTP $($blocked.StatusCode)."
    }
}

$health = & $Executable health_check 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "ReMe health_check failed: $($health -join [Environment]::NewLine)"
}

[pscustomobject]@{
    Address = "127.0.0.1:$Port"
    Version = $version
    ListenerPids = @($owners)
    HealthCheck = ($health -join [Environment]::NewLine)
}
