<#
Registers the Windows task "TMD Radar Weekly Sync" (run once, by the user):
    powershell -File Radar_POC\register_weekly_task.ps1
Every Wednesday 12:00, or at the next start-up if the laptop was off, it runs
weekly_sync_and_test.py with the main checkout's pythonw (no console window).
Remove it with: Unregister-ScheduledTask -TaskName 'TMD Radar Weekly Sync' -Confirm:$false
#>
$ErrorActionPreference = 'Stop'
$name = 'TMD Radar Weekly Sync'
$a = New-ScheduledTaskAction -Execute 'C:\Users\Brandon\Documents\Phase 1\.venv\Scripts\pythonw.exe' `
    -Argument 'Radar_POC\weekly_sync_and_test.py' -WorkingDirectory (Split-Path $PSScriptRoot -Parent)
$t = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Wednesday -At '12:00'
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 3)
Register-ScheduledTask -TaskName $name -Action $a -Trigger $t -Settings $s -Force -Description `
    'Mirrors the cloud TMD radar archive to this laptop and runs the frozen radar forward tests when due (Radar_POC/STACK_TEST_PLAN.md).' | Out-Null
Start-ScheduledTask -TaskName $name          # one run now, as the check
"Registered and started '$name'. In a minute, the last lines of the log should end with 'done:'"
"  Get-Content 'C:\Users\Brandon\Documents\TMD_Radar_Archive_gcp\weekly_run.log' -Tail 8"
