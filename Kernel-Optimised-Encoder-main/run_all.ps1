# PS-09 Windows workflow (PowerShell).
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1                 CPU checks (no NVIDIA GPU needed)
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Precision      + FP16 precision study (CPU emulation, ~30 min)
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -DryRunBench    + official bench.py on 300 passages, CPU FP32
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Report         build docs/technical_report + PDF from results
# If this machine HAS a CUDA GPU, the GPU workflow is run_all.sh (Git Bash): it is not duplicated here.
# Stage 0 (FP32, pad 512) is never run on the 100k set: the reference baseline is a 5,000-passage sample (see README).
# Re-running is safe: finished steps are skipped.
param([switch]$Precision, [switch]$DryRunBench, [switch]$Report)
$ErrorActionPreference = "Continue"   # native-tool exit codes are checked explicitly (PS 5.1 turns stderr into errors under "Stop")
Set-Location $PSScriptRoot
$R = "results_cpu"
New-Item -ItemType Directory -Force $R | Out-Null
$failed = @()
function Step($name, [scriptblock]$body) {
  Write-Host "== $name" -ForegroundColor Cyan
  & $body
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name (exit $LASTEXITCODE)" -ForegroundColor Red; $script:failed += $name }
  $global:LASTEXITCODE = 0
}

# ---- Python: a real interpreter (the Microsoft Store 'python' alias is only an installer stub)
$base = $null
foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")) {
  if (Test-Path $c) { $base = $c; break }
}
if (-not $base) { try { $v = & py -3 --version 2>$null; if ($LASTEXITCODE -eq 0) { $base = "py" } } catch {} }
if (-not $base -and -not (Test-Path .venv)) {
  throw "No Python found. Install it for your user only:  winget install --id Python.Python.3.12 --scope user"
}
if (-not (Test-Path .venv)) { if ($base -eq "py") { & py -3 -m venv .venv } else { & $base -m venv .venv } }
$py = ".\.venv\Scripts\python.exe"
& $py -c "import torch, transformers, numpy, matplotlib, markdown" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
  & $py -m pip install --upgrade pip
  & $py -m pip install -r requirements.txt
}
$env:PYTHONWARNINGS = "ignore"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

& $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -eq 0) {
  Write-Host "CUDA GPU found: running the GPU workflow (run_all.sh) through Git Bash" -ForegroundColor Yellow
  bash run_all.sh; exit $LASTEXITCODE
}
Write-Host "No CUDA GPU: CPU development checks only (GPU numbers come from the Ubuntu machine)" -ForegroundColor Yellow

Step "1. Environment report" { & $py scripts\env_report.py | Out-File -Encoding utf8 "$R\env.json"; Get-Content "$R\env.json" }

Step "2. Dataset" {
  if (-not (Test-Path data\practice_passages.jsonl)) {
    & $py -c "from huggingface_hub import snapshot_download; snapshot_download('Tarakeshwaran/CIT_Hackathon_PS09_encoder', repo_type='dataset', local_dir='data')"
  } else { Write-Host "data\ present" }
}

Step "3. Installer is in sync with the canonical files" { & $py scripts\build_installer.py --check }

Step "4. Edge-case correctness vs reference.py (CPU FP32, stages 0 2 4)" {
  if (-not (Test-Path "$R\test_correctness.json")) { & $py tests\test_correctness.py --data data --out "$R\test_correctness.json" }
  else { Write-Host "done earlier: $R\test_correctness.json" }
}

Step "4b. FP16 and FP16+FP32-residual code paths (native CPU FP16, worst-case passages)" {
  if (-not (Test-Path "$R\test_lowprec_cpu.json")) { & $py tests\test_lowprec_cpu.py --data data --out "$R\test_lowprec_cpu.json" }
  else { Write-Host "done earlier" }
}

Step "5. Accuracy gate, default config on CPU (300 passages + 100 queries)" {
  if (-not (Test-Path "$R\validate_default_300.json")) { & $py validate.py --data data --limit 300 --queries 100 --out "$R\validate_default_300.json" }
  else { Write-Host "done earlier" }
}

Step "6. Workload / padding analysis (exact, tokenizer only)" {
  if (-not (Test-Path "$R\workload_stats.json")) { & $py scripts\workload_stats.py --data data --out "$R\workload_stats.json" }
  else { Write-Host "done earlier" }
}

if ($Precision) {
  Step "7. FP16 precision study (CPU emulation)" {
    if (-not (Test-Path "$R\precision_study.json")) { & $py scripts\precision_study.py --data data --out "$R\precision_study.json" }
    else { Write-Host "done earlier" }
  }
}

if ($DryRunBench) {
  Step "8. Official bench.py dry run on CPU (300 passages, FP32; NOT a GPU score)" {
    $env:PS09_TOKEN_BUDGET = "8192"
    if (-not (Test-Path "$R\ref_throughput.json")) {
      & $py data\reference.py --passages data\practice_passages.jsonl --out "$R\ref300.npy" --limit 300 | Select-Object -Last 1 | Out-File -Encoding ascii "$R\ref_throughput.json"
    }
    & $py run_ablation.py --data data --stages 0 2 4 --limit 300 --no-trace --out-dir "$R\bench_dryrun" --ref-throughput "$R\ref_throughput.json" --tag "CPU dry run"
    Remove-Item Env:PS09_TOKEN_BUDGET
  }
}

if ($Report) { Step "9. Technical report (Markdown + HTML + PDF)" { & $py scripts\make_report.py } }

if ($failed.Count) { Write-Host "`nFAILED steps: $($failed -join '; ')" -ForegroundColor Red; exit 1 }
Write-Host "`nAll requested CPU steps passed. Results: $R" -ForegroundColor Green
