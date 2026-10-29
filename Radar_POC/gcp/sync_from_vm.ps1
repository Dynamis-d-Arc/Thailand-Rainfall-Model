<#
Pull new radar files from the Google Cloud collector VM into a local mirror (Phase 4 POC).

Copies the VM's manifest and logs, works out which archived files are not here yet, and
fetches only those as one tar, so each sync spends the 1 GB/month free egress on new data only.
Safe to re-run: an interrupted sync just leaves files to fetch next time.

Usage (from the worktree root):
    powershell -File Radar_POC\gcp\sync_from_vm.ps1 -VmHost 34.82.1.2
Then report on the mirror:
    $env:TMD_RADAR_ARCHIVE = 'C:\Users\Brandon\Documents\TMD_Radar_Archive_gcp'
    & 'C:\Users\Brandon\Documents\Phase 1\.venv\Scripts\python.exe' Radar_POC\tmd_radar_collect.py report
#>
param(
    [Parameter(Mandatory = $true)] [string]$VmHost,
    [string]$User = 'brandon',
    [string]$Key = "$HOME\.ssh\tmd_radar_vm",
    [string]$Local = 'C:\Users\Brandon\Documents\TMD_Radar_Archive_gcp'
)
$ErrorActionPreference = 'Stop'
$remote = "$User@$VmHost"
$opts = @('-i', $Key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new')

function Invoke-Native([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "$exe exited $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force $Local | Out-Null
Invoke-Native scp ($opts + @("${remote}:tmd_radar_archive/manifest.csv", "${remote}:tmd_radar_archive/collection_log.csv",
                             "${remote}:tmd_radar_archive/collector.log", $Local))

$missing = @(Import-Csv (Join-Path $Local 'manifest.csv') | ForEach-Object { $_.path } |
             Where-Object { -not (Test-Path (Join-Path $Local $_)) })
if ($missing.Count -eq 0) {
    'Mirror already up to date.'
} else {
    # LF endings and no BOM, or tar on the VM reads "\r" into every file name
    $list = Join-Path $env:TEMP 'tmd_radar_missing.txt'
    [IO.File]::WriteAllText($list, ($missing -join "`n") + "`n")
    $bundle = Join-Path $Local 'sync_bundle.tar'
    Invoke-Native scp ($opts + @($list, "${remote}:tmd_radar_missing.txt"))
    Invoke-Native ssh ($opts + @($remote, 'cd tmd_radar_archive && tar -cf ~/tmd_radar_bundle.tar -T ~/tmd_radar_missing.txt'))
    Invoke-Native scp ($opts + @("${remote}:tmd_radar_bundle.tar", $bundle))
    Invoke-Native tar @('-xf', $bundle, '-C', $Local)
    Remove-Item $bundle, $list
    Invoke-Native ssh ($opts + @($remote, 'rm -f ~/tmd_radar_bundle.tar ~/tmd_radar_missing.txt'))
    "Fetched $($missing.Count) new files."
}
'Last VM runs:'
Get-Content (Join-Path $Local 'collector.log') -Tail 3
