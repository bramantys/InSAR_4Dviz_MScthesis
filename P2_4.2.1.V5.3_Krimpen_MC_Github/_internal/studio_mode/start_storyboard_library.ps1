param(
    [int]$Port = 5511
)

$ErrorActionPreference = "Stop"
$serverScript = Join-Path $PSScriptRoot "storyboard_library_server.py"
if (-not (Test-Path $serverScript)) {
    throw "Storyboard library server not found: $serverScript"
}

$pythonCommand = Get-Command py -ErrorAction SilentlyContinue
if ($pythonCommand) {
    & $pythonCommand.Source $serverScript --port $Port --root $PSScriptRoot
    exit $LASTEXITCODE
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    & $pythonCommand.Source $serverScript --port $Port --root $PSScriptRoot
    exit $LASTEXITCODE
}

throw "Python was not found. Install Python or make 'py' / 'python' available in PATH."
