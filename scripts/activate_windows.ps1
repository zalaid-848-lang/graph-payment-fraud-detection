$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$taskTemp = Join-Path $projectRoot ".tmp"
$taskPipCache = Join-Path $projectRoot ".pip-cache"
$activateScript = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"

New-Item -ItemType Directory -Force -Path $taskTemp, $taskPipCache | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
$env:PIP_CACHE_DIR = $taskPipCache

if (-not (Test-Path -LiteralPath $activateScript)) {
    throw "Virtual environment not found. Create .venv before running this script."
}

. $activateScript

Write-Host "Activated project environment: $projectRoot"
Write-Host "Temporary files and pip cache: $taskTemp"
