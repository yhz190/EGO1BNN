# build.ps1 -- firmware build: C/S -> elf -> bin -> hex (for $readmemh)
$ErrorActionPreference = 'Stop'
$RV = if ($env:RISCV_GCC_BIN) { $env:RISCV_GCC_BIN } else { $null }
if (-not $RV) {
    $c = Get-Command riscv32-esp-elf-gcc.exe -ErrorAction SilentlyContinue
    if ($c) { $RV = Split-Path $c.Source -Parent }
}
if (-not $RV) { throw 'riscv32-esp-elf-gcc.exe not found; set RISCV_GCC_BIN' }
$GCC  = Join-Path $RV 'riscv32-esp-elf-gcc.exe'
$OC   = Join-Path $RV 'riscv32-esp-elf-objcopy.exe'
$OD   = Join-Path $RV 'riscv32-esp-elf-objdump.exe'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$PY   = Join-Path $repoRoot '.venv\Scripts\python.exe'
$B2H  = Join-Path $repoRoot 'src/pc\bin2hex.py'
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE

$gccArgs = @(
  '-march=rv32i', '-mabi=ilp32', '-Os', '-nostdlib', '-nostartfiles', '-ffreestanding',
  '-T', 'link.ld', '-Wl,--build-id=none', '-Wl,--no-warn-rwx-segments',
  '-o', 'fw.elf', 'start.S', 'main.c'
)

Write-Host '[build] gcc' -ForegroundColor Cyan
& $GCC @gccArgs
if ($LASTEXITCODE -ne 0) { throw 'gcc failed' }

Write-Host '[build] objcopy' -ForegroundColor Cyan
& $OC -O binary fw.elf fw.bin

Write-Host '[build] bin2hex' -ForegroundColor Cyan
& $PY $B2H (Join-Path $HERE 'fw.bin') (Join-Path $HERE 'fw.hex')

Write-Host '[check] custom-0 instructions' -ForegroundColor Cyan
$dis = & $OD -d fw.elf
$n = 0
foreach ($line in $dis) {
    $m = [regex]::Match($line, '^\s+([0-9a-f]+):\s+([0-9a-f]{8})\s')
    if ($m.Success) {
        $w = [Convert]::ToUInt32($m.Groups[2].Value, 16)
        if (($w -band 0x7F) -eq 0x0B) { $n++; Write-Host ('    @{0} 0x{1}' -f $m.Groups[1].Value, $m.Groups[2].Value) }
    }
}
Write-Host ('    total = {0}' -f $n)
Write-Host '[done] artifacts' -ForegroundColor Cyan
Get-ChildItem fw.elf, fw.bin, fw.hex | ForEach-Object { '    {0,-8} {1,6:N0} B' -f $_.Name, $_.Length }