# run_bench.ps1 — Build + run ter_cuda_forward_packed_v12 on Windows.
# Imports MSVC env from vcvars64.bat, then calls the bench via msys64 bash.
$ErrorActionPreference = 'Stop'

$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$Bash   = 'C:\msys64\usr\bin\bash.exe'

if (-not (Test-Path $VcVars)) { Write-Error "vcvars64 not found: $VcVars"; exit 1 }
if (-not (Test-Path $Bash))   { Write-Error "msys64 bash not found: $Bash";  exit 1 }

# Import vcvars env into current PowerShell session.
$envDump = cmd /c "`"$VcVars`" >nul 2>&1 && set"
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}

# Sanity print to stderr.
$cl   = (Get-Command cl.exe   -ErrorAction SilentlyContinue).Source
$nvcc = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
[Console]::Error.WriteLine("[run_bench] CL   = $cl")
[Console]::Error.WriteLine("[run_bench] NVCC = $nvcc")
if (-not $cl)   { Write-Error "cl.exe not in PATH after vcvars64"; exit 1 }
if (-not $nvcc) { Write-Error "nvcc.exe not in PATH";              exit 1 }

# Run the bench. msys64 bash often refuses to inherit Windows PATH cleanly,
# so we pass the absolute nvcc path via the NVCC env var (bench_v12.sh honors
# `${NVCC:-nvcc}`). Also pass the MSVC bin dir so nvcc finds cl.exe.
$env:NVCC = $nvcc
# Concatenate Windows PATH with msys64's own; explicit-quote semicolon.
$env:PATH = "$($env:PATH);" + (Split-Path $cl)

# Skip bash entirely — build + run the bench directly from PowerShell.
$CudaDir = 'C:\Users\xasko\ter\cuda'
$Src     = Join-Path $CudaDir 'ter_cuda_forward_packed_v12.cu'
$Bin     = Join-Path $CudaDir 'ter_cuda_forward_packed_v12.exe'
$Arch    = if ($env:ARCH)   { $env:ARCH   } else { 'sm_86' }
$Iters   = if ($env:ITERS)  { $env:ITERS  } else { '200' }
$Warmup  = if ($env:WARMUP) { $env:WARMUP } else { '20' }

if (-not (Test-Path $Src)) { Write-Error "source missing: $Src"; exit 1 }

[Console]::Error.WriteLine("[run_bench] Compiling for $Arch ...")
$nvccArgs = @('-O3', '-std=c++17', "-arch=$Arch", '-Xptxas', '-O3', $Src, '-o', $Bin)
$buildLog = & $nvcc @nvccArgs 2>&1
$buildExit = $LASTEXITCODE
$buildLog | ForEach-Object { [Console]::Error.WriteLine("[nvcc] $_") }
if ($buildExit -ne 0) { Write-Error "nvcc build failed ($buildExit)"; exit $buildExit }
if (-not (Test-Path $Bin)) { Write-Error "binary missing after build: $Bin"; exit 1 }

[Console]::Error.WriteLine("[run_bench] Running $Bin iters=$Iters warmup=$Warmup")
# CSV to stdout; let it flow through.
& $Bin $Iters $Warmup
exit $LASTEXITCODE
