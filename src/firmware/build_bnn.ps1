# build_bnn.ps1 -- top_bnn firmware build: main_bnn.c -> fw.elf / fw.bin / fw.hex
#
#   fw.hex  : 32-bit little-endian words, one per line, for $readmemh into IMEM
#   outputs : src/firmware/bnn/  (the D3 artefacts in src/firmware/ are left untouched)
#
# No python / venv dependency: the bin -> hex step is done in PowerShell.
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
$SZ   = Join-Path $RV 'riscv32-esp-elf-size.exe'
$NM   = Join-Path $RV 'riscv32-esp-elf-nm.exe'

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$OUT  = Join-Path $HERE 'bnn'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
Set-Location $HERE

$gccArgs = @(
  '-march=rv32i', '-mabi=ilp32', '-Os', '-nostdlib', '-nostartfiles', '-ffreestanding',
  '-T', 'link_bnn.ld', '-Wl,--build-id=none', '-Wl,--no-warn-rwx-segments',
  '-o', (Join-Path $OUT 'fw.elf'), 'start.S', 'main_bnn.c'
)

Write-Host '[build] gcc' -ForegroundColor Cyan
& $GCC @gccArgs
if ($LASTEXITCODE -ne 0) { throw 'gcc failed' }

Write-Host '[build] objcopy' -ForegroundColor Cyan
& $OC -O binary (Join-Path $OUT 'fw.elf') (Join-Path $OUT 'fw.bin')
if ($LASTEXITCODE -ne 0) { throw 'objcopy failed' }

Write-Host '[build] bin -> hex' -ForegroundColor Cyan
$bin = [System.IO.File]::ReadAllBytes((Join-Path $OUT 'fw.bin'))
if (($bin.Length % 4) -ne 0) {
  $pad = New-Object byte[] (4 - ($bin.Length % 4))
  $bin = $bin + $pad
}
$nw = $bin.Length / 4
$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt $nw; $i++) {
  $w = [uint32]$bin[$i*4] -bor ([uint32]$bin[$i*4+1] -shl 8) -bor ([uint32]$bin[$i*4+2] -shl 16) -bor ([uint32]$bin[$i*4+3] -shl 24)
  [void]$sb.AppendFormat("{0:x8}`n", $w)
}
[System.IO.File]::WriteAllText((Join-Path $OUT 'fw.hex'), $sb.ToString())

Write-Host '[check] section sizes (IMEM must hold .text/.rodata, .data/.bss empty)' -ForegroundColor Cyan
& $SZ (Join-Path $OUT 'fw.elf')

Write-Host '[check] Custom-0 instructions in the image' -ForegroundColor Cyan
$dis = & $OD -d (Join-Path $OUT 'fw.elf')
$n = 0
foreach ($line in $dis) {
  $m = [regex]::Match($line, '^\s+([0-9a-f]+):\s+([0-9a-f]{8})\s')
  if ($m.Success) {
    $w = [Convert]::ToUInt32($m.Groups[2].Value, 16)
    if (($w -band 0x7F) -eq 0x0B) {
      $n++
      $f3 = ($w -shr 12) -band 7
      $f7 = ($w -shr 25) -band 0x7F
      Write-Host ("    @{0} 0x{1}  funct3={2} funct7={3}" -f $m.Groups[1].Value, $m.Groups[2].Value, $f3, $f7)
    }
  }
}
Write-Host ('    total = {0} Custom-0 instructions' -f $n)
if ($n -lt 4) { Write-Host '[warn] expected at least 4 Custom-0 instructions (conv/wait/acc x4 + illegal)' }

Write-Host ('[done] {0} words = {1} bytes of IMEM image' -f $nw, $bin.Length) -ForegroundColor Cyan
Get-ChildItem (Join-Path $OUT 'fw.elf'), (Join-Path $OUT 'fw.bin'), (Join-Path $OUT 'fw.hex') |
  ForEach-Object { '    {0,-8} {1,6:N0} B  {2}' -f $_.Name, $_.Length, $_.FullName }
