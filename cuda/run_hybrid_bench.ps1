$ErrorActionPreference = 'Stop'
$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$envDump = cmd /c "`"$VcVars`" >nul 2>&1 && set"
foreach ($line in $envDump) { if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] } }
$nvcc = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
$Dir = 'C:\Users\xasko\ter\cuda'
$Src = Join-Path $Dir 'ter_cuda_hybrid_bench.cu'
$Bin = Join-Path $Dir 'ter_cuda_hybrid_bench.exe'
[Console]::Error.WriteLine("[hybrid] Compiling ...")
$nvccArgs = @('-O3', '-std=c++17', '-arch=sm_86', '-Xptxas', '-O3', $Src, '-o', $Bin)
$buildLog = & $nvcc @nvccArgs 2>&1
$buildExit = $LASTEXITCODE
$buildLog | ForEach-Object { [Console]::Error.WriteLine("[nvcc] $_") }
if ($buildExit -ne 0) { Write-Error "build failed ($buildExit)"; exit $buildExit }
[Console]::Error.WriteLine("[hybrid] Running ...")
$Iters  = if ($env:ITERS)  { $env:ITERS  } else { '200' }
$Warmup = if ($env:WARMUP) { $env:WARMUP } else { '20' }
& $Bin $Iters $Warmup
exit $LASTEXITCODE
