param([string]$OutputDirectory = '')
$ErrorActionPreference = 'Stop'
$compiler = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$sim = (Resolve-Path "$PSScriptRoot/../../../Simulator").Path
$bin = if ($OutputDirectory) { [IO.Path]::GetFullPath($OutputDirectory) } else { Join-Path $PSScriptRoot 'bin' }
New-Item -ItemType Directory -Force $bin | Out-Null
$dependencies = Join-Path $PSScriptRoot 'ilspy'
if (!(Test-Path -LiteralPath (Join-Path $dependencies 'Mono.Cecil.dll'))) {
    $archive = Join-Path $PSScriptRoot 'ilspy.zip'
    Invoke-WebRequest 'https://github.com/icsharpcode/ILSpy/releases/download/v7.2/ILSpy_binaries_7.2.0.6844.zip' -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne '61341AEB5992BC76ECD09F29D4A39D13D96AFB4D3A498FC191FD999908EA667C') { throw 'Unexpected build dependency archive hash' }
    Expand-Archive -LiteralPath $archive -DestinationPath $dependencies -Force
}
$patcher = Join-Path $dependencies 'PatchTransport.exe'
& $compiler /nologo ("/out:" + $patcher) ("/r:" + (Join-Path $dependencies 'Mono.Cecil.dll')) (Join-Path $PSScriptRoot 'PatchTransport.cs')
if($LASTEXITCODE -ne 0){throw 'Transport patcher build failed'}
& $patcher (Join-Path $sim 'Pinokio.TCP.IP.dll') (Join-Path $bin 'Pinokio.TCP.IP.dll')
if($LASTEXITCODE -ne 0){throw 'Transport compatibility check failed'}
# Keep native log files in each episode runtime, not in the GUI's Simulator/Log.
Copy-Item -LiteralPath (Join-Path $sim 'Pinokio.Utill.Log.dll') -Destination $bin
$refs = @('Simulation.Engine','Simulation.Model','Pinokio.Sim.Definition','Pinokio.Sim.Interface','Pinokio.CAP.Data','Pinokio.CAP.Manager','Pinokio.CAP.DeepLearning','Pinokio.TCP.IP','Pinokio.Utill.Log','AMHS.OSS.Logic') | ForEach-Object { '/r:' + (Join-Path $sim ($_ + '.dll')) }
& $compiler /nologo /optimize+ /platform:x64 ("/out:" + (Join-Path $bin "Pinokio.Headless.exe")) @refs /r:System.Windows.Forms.dll /r:System.Data.dll /r:System.Drawing.dll (Join-Path $PSScriptRoot "HeadlessProgram.cs") (Join-Path $PSScriptRoot "HeadlessLoader.cs")
if($LASTEXITCODE -ne 0){throw 'Headless build failed'}
$frameworkVerifier = Join-Path $dependencies 'VerifyFramework.exe'
& $compiler /nologo ("/out:" + $frameworkVerifier) ("/r:" + (Join-Path $dependencies 'Mono.Cecil.dll')) (Join-Path $PSScriptRoot 'VerifyFramework.cs')
if($LASTEXITCODE -ne 0){throw 'Framework verifier build failed'}
& $frameworkVerifier (Join-Path $sim 'Pinokio.Simulator.exe') (Join-Path $bin 'Pinokio.Headless.exe')
if($LASTEXITCODE -ne 0){throw 'GUI/headless framework compatibility check failed'}
Copy-Item -LiteralPath (Join-Path $sim 'Pinokio.Simulator.exe.config') -Destination (Join-Path $bin 'Pinokio.Headless.exe.config')

[xml]$cfg = Get-Content (Join-Path $bin 'Pinokio.Headless.exe.config')
$load = $cfg.CreateElement('loadFromRemoteSources')
$load.SetAttribute('enabled', 'true')
$cfg.configuration.runtime.AppendChild($load) | Out-Null
$cfg.Save((Join-Path $bin 'Pinokio.Headless.exe.config'))
