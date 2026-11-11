param(
    [string]$Python = 'D:\ShiftLite-envs\cpu-312\python.exe',
    [string]$RunRoot = 'D:\ShiftLite-runs\t0-t1-2026-11-09',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @('serve')
)
# 运行副本保存模型/数据/本机结果，源仓库保留伙伴的历史产物。
$ErrorActionPreference = 'Stop'
$source = $PSScriptRoot
$target = Join-Path $RunRoot 'impl'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python 不存在: $Python" }
if ([IO.Path]::GetFullPath($source).TrimEnd('\') -eq [IO.Path]::GetFullPath($target).TrimEnd('\')) {
    throw '运行目录不能与源目录相同。'
}
New-Item -ItemType Directory -Path $target -Force | Out-Null
foreach ($name in @('core', 'web', 'tests')) {
    New-Item -ItemType Directory -Path (Join-Path $target $name) -Force | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $source $name) -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path (Join-Path $target $name) $_.Name)
    }
}
foreach ($name in @('cli.py', 'server.py', 'requirements.txt')) {
    Copy-Item -LiteralPath (Join-Path $source $name) -Destination (Join-Path $target $name)
}
$env:TORCH_HOME = 'D:\ShiftLite-cache\torch'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $target
try { & $Python -B -X utf8 cli.py @CliArgs; $result = $LASTEXITCODE }
finally { Pop-Location }
exit $result
