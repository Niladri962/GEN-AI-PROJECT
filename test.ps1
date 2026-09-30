param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Project environment not found. Create it with 'py -3.10 -m venv .venv' and install dependencies with '.\.venv\Scripts\python.exe -m pip install -r requirements.txt'."
    exit 1
}

& $Python -m pytest @PytestArgs
exit $LASTEXITCODE
