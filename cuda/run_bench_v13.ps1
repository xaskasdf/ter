# run_bench_v13.ps1 — Build + run v13 bench (5 kernels × 5 shapes).
$ErrorActionPreference = 'Stop'

$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $VcVars)) { Write-Error "vcvars64 missing"; exit 1 }

# Import vcvars env into this PowerShell session.
$envDump = cmd /c "`"$VcVars`" >nul 2>&1 && set"
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}

$cl   = (Get-Command cl.exe   -ErrorAction SilentlyContinue).Source
$nvcc = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
[Console]::Error.WriteLine("[v13] CL   = $cl")
[Console]::Error.WriteLine("[v13] NVCC = $nvcc")
if (-not $cl -or -not $nvcc) { Write-Error "missing toolchain"; exit 1 }

$CudaDir = 'C:\Users\xasko\ter\cuda'
$Src     = Join-Path $CudaDir 'ter_cuda_forward_packed_v13_bench.cu'
$Bin     = Join-Path $CudaDir 'ter_cuda_forward_packed_v13_bench.exe'
$Arch    = if ($env:ARCH)   { $env:ARCH   } else { 'sm_86' }
$Iters   = if ($env:ITERS)  { $env:ITERS  } else { '200' }
$Warmup  = if ($env:WARMUP) { $env:WARMUP } else { '20' }

[Console]::Error.WriteLine("[v13] Compiling for $Arch ...")
$nvccArgs = @('-O3', '-std=c++17', "-arch=$Arch", '-Xptxas', '-O3', $Src, '-o', $Bin)
$buildLog = & $nvcc @nvccArgs 2>&1
$buildExit = $LASTEXITCODE
$buildLog | ForEach-Object { [Console]::Error.WriteLine("[nvcc] $_") }
if ($buildExit -ne 0) { Write-Error "nvcc build failed ($buildExit)"; exit $buildExit }

[Console]::Error.WriteLine("[v13] Running $Bin iters=$Iters warmup=$Warmup")
& $Bin $Iters $Warmup
exit $LASTEXITCODE
