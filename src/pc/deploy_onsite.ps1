# deploy_onsite.ps1 -- 把现场标定好的权重烧进板子（只换权重，RTL/固件不动）
# 用法： powershell -NoProfile -ExecutionPolicy Bypass -File deploy_onsite.ps1 `
#            -BlobDir <repo>\board/weights\hw_onsite `
#            -Ckpt   <repo>\board/weights\ONSITE_DA.pt
param(
  [Parameter(Mandatory=$true)][string]$BlobDir,
  [Parameter(Mandatory=$true)][string]$Ckpt,
  [string]$Port = 'COM6',
  [int]$Baud = 921600
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$build = if ($env:EGO1_BUILD_DIR) { $env:EGO1_BUILD_DIR } else { Join-Path $repoRoot '..\build_top' }
$vivado = if ($env:VIVADO_BAT) { $env:VIVADO_BAT } else { 'vivado' }
$py = Join-Path $root '.venv\Scripts\python.exe'

Write-Host "[deploy] blob = $BlobDir"
if (-not (Test-Path (Join-Path $BlobDir 'weights.hex'))) { throw "weights.hex not found in $BlobDir" }
$lines = (Get-Content (Join-Path $BlobDir 'weights.hex') | Measure-Object).Count
if ($lines -ne 271) { throw "weights.hex has $lines lines, expected 271 (v5 beat-major)" }

Write-Host '[deploy] refresh build tree (RTL + firmware unchanged, only weights)'
$src = Join-Path $build 'src'
Remove-Item (Join-Path $src '*.v') -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root 'src/rtl\*.v') $src -Force
Copy-Item (Join-Path $root 'src/rtl\bnn\*.v') $src -Force
Copy-Item (Join-Path $root 'src/rtl\picorv32\*.v') $src -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root 'board\board_ego1_top_bnn.xdc') $src -Force
Copy-Item (Join-Path $root 'src/firmware\bnn\fw.hex') (Join-Path $build 'fw.hex') -Force
Copy-Item (Join-Path $BlobDir 'weights.hex') (Join-Path $build 'weights.hex') -Force

Write-Host '[deploy] vivado synth + impl + bitstream (~7 min)'
Push-Location $build
& $vivado -mode batch -source (Join-Path $build 'build_top_ascii.tcl') -nojournal -nolog -tclargs impl 2>&1 |
  Select-String -Pattern 'GATE_|BUILD_TOP_DONE|ERROR' | ForEach-Object { '  ' + $_.Line }
Pop-Location
$bit = Join-Path $build 'patch2\patch2.runs\impl_1\top_bnn.bit'
if (-not (Test-Path $bit)) { throw 'bitstream not generated' }

Write-Host '[deploy] program FPGA'
$tcl = @"
open_hw_manager
connect_hw_server -url localhost:3121 -allow_non_jtag
open_hw_target [lindex [get_hw_targets] 0]
set dev [lindex [get_hw_devices] 0]
current_hw_device `$dev
set_property PROGRAM.FILE {$($bit -replace '\\','/')} `$dev
program_hw_devices `$dev
puts "PROGRAM_OK"
close_hw_target
disconnect_hw_server
"@
$tmp = Join-Path $build 'hw_program_onsite.tcl'
Set-Content -Path $tmp -Value $tcl -Encoding UTF8
& $vivado -mode batch -source $tmp -nojournal -nolog 2>&1 | Select-String 'PROGRAM_OK|ERROR' | ForEach-Object { '  ' + $_.Line }

Write-Host '[deploy] board self-test on the onsite held-out frames'
# prefer the bins written by onsite_train_aug.py (v5/adapt flow), fall back to legacy path
$binsV5 = Join-Path $root 'sim\onsite_aug_test_bins'
$bins = if (Test-Path $binsV5) { $binsV5 } else { Join-Path $root 'sim\onsite_test_bins' }
if (Test-Path $bins) {
  Push-Location (Join-Path $root 'src/pc')
  & $py board_smoke.py $Port $Baud 200 --dir $bins --ckpt $Ckpt 2>&1 | Select-String 'RESULT|elapsed|throughput|DIFF' | ForEach-Object { '  ' + $_.Line }
  Pop-Location
} else {
  Write-Host '  (no onsite_test_bins; skipping board self-test)'
}

# archive
$dst = Join-Path $root 'report\ego1_bnn_onsite.bit'
Copy-Item $bit $dst -Force
$md5 = (Get-FileHash $dst -Algorithm MD5).Hash.ToLower()
Write-Host "[deploy] done: $dst  md5=$md5"
