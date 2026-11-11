param(
    [string]$Python,
    [string]$RunRoot,
    [string]$TorchHome,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @('serve')
)
# 运行副本保存模型/数据/本机结果，源仓库保留伙伴的历史产物。
$ErrorActionPreference = 'Stop'
$source = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$repoPath = Split-Path -Parent $source
if (-not $Python) {
    $venvPython = Join-Path $repoPath '.venv\Scripts\python.exe'
    $Python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
}
$pythonCommand = Get-Command -Name $Python -CommandType Application -ErrorAction Stop
if (-not $RunRoot) {
    $RunRoot = Join-Path (Split-Path -Parent $repoPath) ((Split-Path -Leaf $repoPath) + '-runs\local')
}
$RunRoot = [IO.Path]::GetFullPath($RunRoot)
$target = [IO.Path]::GetFullPath((Join-Path $RunRoot 'impl')).TrimEnd('\')
if ($target.Equals($repoPath, [StringComparison]::OrdinalIgnoreCase) -or
    $target.StartsWith($repoPath + '\', [StringComparison]::OrdinalIgnoreCase) -or
    $source.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw '运行副本必须位于源仓库之外，且不能包含源代码目录。'
}
if (-not $TorchHome) {
    $TorchHome = if ($env:TORCH_HOME) { $env:TORCH_HOME } else { Join-Path $RunRoot 'cache\torch' }
}
$TorchHome = [IO.Path]::GetFullPath($TorchHome)
New-Item -ItemType Directory -Path $target -Force | Out-Null
foreach ($name in @('core', 'web', 'tests')) {
    New-Item -ItemType Directory -Path (Join-Path $target $name) -Force | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $source $name) -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path (Join-Path $target $name) $_.Name)
    }
}
foreach ($name in @('cli.py', 'server.py', 'README.md', 'requirements.txt', 'requirements-cpu-win-py312.lock.txt')) {
    Copy-Item -LiteralPath (Join-Path $source $name) -Destination (Join-Path $target $name)
}
$previousTorchHome = $env:TORCH_HOME
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:TORCH_HOME = $TorchHome
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $target
try { & $pythonCommand.Source -B -X utf8 cli.py @CliArgs; $result = $LASTEXITCODE }
finally {
    Pop-Location
    $env:TORCH_HOME = $previousTorchHome
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
}
exit $result
