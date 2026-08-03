#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$LocalGameRoot = 'C:\stg-win-demo',
    [string]$Scenario = 'okuu:Lunatic',
    [ValidateRange(1, 9999)]
    [int]$Attack = 3,
    [int]$Seed = 20260730,
    [string]$Player = 'reimu_player',
    [ValidateRange(1, 65535)]
    [int]$Port = 24816,
    [ValidateRange(1, 86400)]
    [int]$MaxFrames = 4200,
    [ValidateRange(1, 600)]
    [int]$HorizonFrames = 60,
    [ValidateRange(0, 600)]
    [int]$ObservationDelay = 5,
    [ValidateRange(1, 3600)]
    [int]$StartupTimeout = 120,
    [string]$RegionDynamicsMemory = '',
    [string]$OutputPath = '',
    [string]$ReplayName = '',
    [switch]$RecordObservations,
    [switch]$Headless,
    [switch]$DisableOverlay,
    [switch]$CloseGameWhenDone
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ListeningProcessId {
    param([int]$LocalPort)

    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        try {
            $connections = @(Get-NetTCPConnection `
                -LocalPort $LocalPort `
                -State Listen `
                -ErrorAction Stop)
            if ($connections.Count -gt 0) {
                return [int]$connections[0].OwningProcess
            }
            return $null
        }
        catch {
            # Fall through to netstat when NetTCPIP/CIM is unavailable.
        }
    }

    $netstat = Join-Path $env:SystemRoot 'System32\netstat.exe'
    $pattern = '^\s*TCP\s+\S+:' + $LocalPort + '\s+\S+\s+LISTENING\s+(\d+)\s*$'
    foreach ($line in & $netstat -ano -p tcp) {
        if ($line -match $pattern) {
            return [int]$Matches[1]
        }
    }
    return $null
}

function Write-EngineLogTail {
    param([string]$GameRoot)

    $engineLog = Join-Path $GameRoot 'engine.log'
    if (Test-Path -LiteralPath $engineLog -PathType Leaf) {
        Write-Host ''
        Write-Host 'Last 60 lines of engine.log:' -ForegroundColor Yellow
        Get-Content -LiteralPath $engineLog -Tail 60 -ErrorAction SilentlyContinue
    }
}

function Restore-ProcessEnvironment {
    param(
        [string[]]$Names,
        [hashtable]$SavedValues
    )

    foreach ($name in $Names) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $SavedValues[$name],
            [EnvironmentVariableTarget]::Process)
    }
}

$labRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$modRoot = [IO.Path]::GetFullPath((Join-Path $labRoot '..\..'))
$modName = Split-Path -Leaf $modRoot
$gameExe = Join-Path $LocalGameRoot 'LuaSTGSub.exe'
$gameLaunch = Join-Path $LocalGameRoot 'launch'
$localModRoot = Join-Path (Join-Path $LocalGameRoot 'mod') $modName
$sourceBridge = Join-Path $modRoot 'compat\testing\bridge.lua'
$localBridge = Join-Path $localModRoot 'compat\testing\bridge.lua'
$venvPython = Join-Path $labRoot '.venv-win\Scripts\python.exe'

if ([string]::IsNullOrWhiteSpace($RegionDynamicsMemory)) {
    $RegionDynamicsMemory = Join-Path $labRoot 'models\region_dynamics_boss3_v2.json'
}
elseif (-not [IO.Path]::IsPathRooted($RegionDynamicsMemory)) {
    $RegionDynamicsMemory = Join-Path $labRoot $RegionDynamicsMemory
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if ([string]::IsNullOrWhiteSpace($ReplayName)) {
    $ReplayName = "boss3-win-$timestamp"
}
elseif ($ReplayName.EndsWith('.rep', [StringComparison]::OrdinalIgnoreCase)) {
    $ReplayName = $ReplayName.Substring(0, $ReplayName.Length - 4)
}
if ($ReplayName -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$' -or
        $ReplayName.EndsWith('.', [StringComparison]::Ordinal)) {
    throw 'ReplayName must contain 1-96 portable filename characters.'
}
$replayBaseName = $ReplayName.Split('.')[0].ToUpperInvariant()
$reservedReplayNames = @('CON', 'PRN', 'AUX', 'NUL')
if ($replayBaseName -in $reservedReplayNames -or
        $replayBaseName -match '^(COM|LPT)[1-9]$') {
    throw 'ReplayName uses a Windows reserved basename.'
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path `
        (Join-Path $labRoot 'artifacts') `
        ("engine-mpc-boss3-win-{0}.json" -f $timestamp)
}
elseif (-not [IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $labRoot $OutputPath
}
$nativeReplayPath = Join-Path `
    (Join-Path (Join-Path (Join-Path $LocalGameRoot 'userdata') 'replay') $modName) `
    (Join-Path 'analysis' ($ReplayName + '.rep'))

$requiredFiles = @(
    $gameExe,
    $gameLaunch,
    (Join-Path $localModRoot 'root.lua'),
    $sourceBridge,
    $localBridge,
    $venvPython,
    $RegionDynamicsMemory
)
foreach ($path in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}

$sourceBridgeHash = (Get-FileHash -LiteralPath $sourceBridge -Algorithm SHA256).Hash
$localBridgeHash = (Get-FileHash -LiteralPath $localBridge -Algorithm SHA256).Hash
if ($sourceBridgeHash -ne $localBridgeHash) {
    throw @"
The local bridge is not the current non-flickering version.
Source: $sourceBridge
Local:  $localBridge
Close LuaSTG, update the local bridge, and run this script again.
"@
}

$existingGame = @(Get-Process -Name 'LuaSTGSub' -ErrorAction SilentlyContinue)
if ($existingGame.Count -gt 0) {
    $processIds = ($existingGame | ForEach-Object { $_.Id }) -join ', '
    throw "LuaSTGSub is already running (PID: $processIds). Close it before starting the test."
}

$portOwner = Get-ListeningProcessId -LocalPort $Port
if ($null -ne $portOwner) {
    throw "TCP port $Port is already listening in PID $portOwner. Stop that process or select another -Port."
}

$environmentNames = @(
    'PYTHONPATH',
    'SR_TEST_MODE',
    'SR_TEST_HEADLESS',
    'SR_TEST_LOCKSTEP',
    'SR_TEST_PORT',
    'SR_TEST_MAX_FRAMES',
    'SR_TEST_SESSION_ID',
    'SR_TEST_STARTUP_ACCEPT_TIMEOUT',
    'SR_TEST_SOURCE_ROOT',
    'SR_SAFETY_ZONE_OVERLAY'
)
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable(
        $name,
        [EnvironmentVariableTarget]::Process)
}

$gameProcess = $null
$exitCode = 1
try {
    $sourcePath = Join-Path $labRoot 'src'
    if ([string]::IsNullOrWhiteSpace($savedEnvironment['PYTHONPATH'])) {
        $env:PYTHONPATH = $sourcePath
    }
    else {
        $env:PYTHONPATH = $sourcePath + [IO.Path]::PathSeparator + $savedEnvironment['PYTHONPATH']
    }

    $pythonCheckCode = 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14); import numpy, stg_lab; print(sys.version.split()[0])'
    $pythonCheck = & $venvPython -c $pythonCheckCode 2>&1
    $pythonCheckExit = $LASTEXITCODE
    if ($pythonCheckExit -ne 0) {
        throw "The existing .venv-win is unusable: $($pythonCheck -join [Environment]::NewLine)"
    }
    $pythonVersion = ($pythonCheck | Select-Object -Last 1).ToString().Trim()

    $sessionSuffix = [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $sessionId = "win-boss3-$timestamp-$sessionSuffix"
    $env:SR_TEST_MODE = '1'
    $env:SR_TEST_HEADLESS = if ($Headless) { '1' } else { '0' }
    $env:SR_TEST_LOCKSTEP = '1'
    $env:SR_TEST_PORT = $Port.ToString()
    $env:SR_TEST_MAX_FRAMES = $MaxFrames.ToString()
    $env:SR_TEST_SESSION_ID = $sessionId
    $env:SR_TEST_STARTUP_ACCEPT_TIMEOUT = $StartupTimeout.ToString()
    $env:SR_TEST_SOURCE_ROOT = "mod\$modName"
    $env:SR_SAFETY_ZONE_OVERLAY = if ($Headless -or $DisableOverlay) { '0' } else { '1' }

    $gameSettings = @(
        'start_game=true',
        'is_debug=true',
        'setting.nosplash=true',
        'setting.windowed=true',
        'setting.resx=640',
        'setting.resy=480',
        'setting.vsync=true',
        "setting.mod='$modName'",
        'cheat=false',
        'updatelib=false'
    ) -join ' '
    $quotedGameSettings = '"' + $gameSettings + '"'

    Write-Host 'Starting native LuaSTG test...' -ForegroundColor Cyan
    Write-Host "  Game:     $gameExe"
    Write-Host "  Python:   $venvPython ($pythonVersion)"
    Write-Host "  Scenario: $Scenario, attack $Attack, seed $Seed"
    Write-Host "  Session:  $sessionId"
    Write-Host "  Report:   $OutputPath"
    Write-Host "  Replay:   $nativeReplayPath"

    $gameProcess = Start-Process `
        -FilePath $gameExe `
        -WorkingDirectory $LocalGameRoot `
        -ArgumentList $quotedGameSettings `
        -PassThru

    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeout)
    $listeningProcessId = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        $gameProcess.Refresh()
        if ($gameProcess.HasExited) {
            throw "LuaSTGSub exited during startup with code $($gameProcess.ExitCode)."
        }
        $listeningProcessId = Get-ListeningProcessId -LocalPort $Port
        if ($null -ne $listeningProcessId) {
            break
        }
        Start-Sleep -Milliseconds 200
    }

    if ($null -eq $listeningProcessId) {
        throw "LuaSTG did not listen on 127.0.0.1:$Port within $StartupTimeout seconds."
    }
    if ($listeningProcessId -ne $gameProcess.Id) {
        throw "Port $Port belongs to PID $listeningProcessId, not the started LuaSTG PID $($gameProcess.Id)."
    }

    $outputDirectory = Split-Path -Parent $OutputPath
    if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
        $null = New-Item -ItemType Directory -Path $outputDirectory -Force
    }

    [string[]]$controllerArguments = @(
        '-m', 'stg_lab.cli',
        'engine-mpc-play',
        '--host', '127.0.0.1',
        '--port', $Port.ToString(),
        '--timeout', $StartupTimeout.ToString(),
        '--scenario', $Scenario,
        '--attack', $Attack.ToString(),
        '--seed', $Seed.ToString(),
        '--player', $Player,
        '--max-frames', $MaxFrames.ToString(),
        '--horizon-frames', $HorizonFrames.ToString(),
        '--observation-delay', $ObservationDelay.ToString(),
        '--region-dynamics-memory', $RegionDynamicsMemory,
        '--replay-name', $ReplayName,
        '--render-every', '1',
        '--output', $OutputPath
    )
    if ($RecordObservations) {
        $controllerArguments += @('--record-observations-from-frame', '0')
    }
    if ($Headless) {
        $controllerArguments += '--no-render'
    }
    else {
        $controllerArguments += '--render'
    }

    Write-Host "Bridge is listening on 127.0.0.1:$Port. Starting the controller..." -ForegroundColor Green
    & $venvPython @controllerArguments 1> $null
    $controllerExit = $LASTEXITCODE
    if ($controllerExit -ne 0) {
        throw "The test failed strict attack_complete validation (controller exit code $controllerExit). Report: $OutputPath"
    }
    if (-not (Test-Path -LiteralPath $nativeReplayPath -PathType Leaf)) {
        throw "The controller completed but the native replay is missing: $nativeReplayPath"
    }
    if ((Get-Item -LiteralPath $nativeReplayPath).Length -le 0) {
        throw "The controller created an empty native replay: $nativeReplayPath"
    }

    Write-Host ''
    Write-Host 'PASS: the engine reported attack_complete.' -ForegroundColor Green
    Write-Host "Report: $OutputPath"
    Write-Host "Replay: $nativeReplayPath"
    $exitCode = 0
}
catch {
    Write-Host ''
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-EngineLogTail -GameRoot $LocalGameRoot
    $exitCode = 1
}
finally {
    Restore-ProcessEnvironment -Names $environmentNames -SavedValues $savedEnvironment

    if (($CloseGameWhenDone -or $exitCode -ne 0) -and $null -ne $gameProcess) {
        $gameProcess.Refresh()
        if (-not $gameProcess.HasExited) {
            $null = $gameProcess.CloseMainWindow()
            try {
                Wait-Process -Id $gameProcess.Id -Timeout 5 -ErrorAction Stop
            }
            catch {
                Write-Warning 'LuaSTG did not close; close the game window manually.'
            }
        }
    }
}

exit $exitCode
