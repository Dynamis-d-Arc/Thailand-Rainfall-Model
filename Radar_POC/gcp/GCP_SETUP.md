# Run the radar collector on a Google Cloud e2-micro

This moves collection off the laptop so sleep stops costing frames. The VM runs the same
`tmd_radar_collect.py` every 5 min from cron, and `sync_from_vm.ps1` pulls the new files back
to the laptop. Keep the laptop's own collector running as a backup.

## Cost

About **US$3.65 a month**, which is the VM's public IPv4 address at $0.005/h. TMD has no IPv6,
so the address can't be avoided. Everything else is inside the free tier, provided you keep to
these rules:

- **Machine:** one `e2-micro` per billing account. If that account already runs an e2-micro
  (for example in the Earth Engine project), this one is billed in full.
- **Region:** `us-west1`, `us-central1` or `us-east1` only. Oregon (`us-west1`) is the closest
  to Thailand.
- **Disk:** a *standard* persistent disk of 30 GB or less. The console's default "Balanced" disk
  and any snapshot schedule are billed. The command below avoids both.
- **Egress:** 1 GB a month is free. A month of archive is about 0.9 GB, and extra costs about
  $0.12/GB.

If your billing account still has free-trial credit, the credit covers the address too. Set a
budget alert either way (step 2).

## 1. Make an SSH key on the laptop (PowerShell)

```powershell
ssh-keygen -t ed25519 -f "$HOME\.ssh\tmd_radar_vm" -C brandon
Get-Content "$HOME\.ssh\tmd_radar_vm.pub"
```

Press Enter at the passphrase prompt to leave it empty: the sync script runs non-interactively.
The key is used only for this VM. Copy the line that `Get-Content` prints; it starts with
`ssh-ed25519` and ends with `brandon`.

## 2. Project, billing and budget (Google Cloud console, by you)

1. Go to https://console.cloud.google.com and create a project, for example `tmd-radar-poc`.
   Note its **project ID**.
2. Link it to your billing account when the console asks.
3. In **Billing → Budgets & alerts**, create a budget of $5 a month with email alerts.

## 3. Create the VM (Cloud Shell)

Open Cloud Shell with the `>_` icon at the top right of the console, then paste the commands
below. Replace `YOUR_PROJECT_ID`, and replace `PASTE_PUBLIC_KEY_HERE` with the whole line from
step 1.

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com
gcloud compute instances create tmd-radar \
  --zone=us-west1-b --machine-type=e2-micro \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-standard \
  --metadata=ssh-keys="brandon:PASTE_PUBLIC_KEY_HERE"
gcloud compute instances describe tmd-radar --zone=us-west1-b \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

The last command prints the VM's IP address. New projects have a default firewall rule that
allows SSH, so nothing else is needed.

## 4. Install the collector (PowerShell, from the worktree root)

```powershell
cd C:\Users\Brandon\Documents\Phase1_radar_poc
$ip = '34.x.x.x'          # the address from step 3
scp -i "$HOME\.ssh\tmd_radar_vm" Radar_POC\tmd_radar_collect.py Radar_POC\gcp\setup_vm.sh "brandon@${ip}:~/"
ssh -i "$HOME\.ssh\tmd_radar_vm" "brandon@$ip" "bash ~/setup_vm.sh"
```

On the first connection, type `yes` to accept the VM's key. The setup takes a couple of
minutes, and it should end with these three things:

- `selftest ok`;
- one run line like `nc 2026-...Z stored (...); png stored 25 of 25 new, ...`;
- the cron line.

That first run backfills the 6 h of frames TMD has up. Check the first scheduled runs about 15
minutes later:

```powershell
ssh -i "$HOME\.ssh\tmd_radar_vm" "brandon@$ip" "tail -3 ~/tmd_radar_archive/collector.log"
```

## 5. Pull the data back (weekly, or before any analysis)

```powershell
powershell -File Radar_POC\gcp\sync_from_vm.ps1 -VmHost $ip
$env:TMD_RADAR_ARCHIVE = 'C:\Users\Brandon\Documents\TMD_Radar_Archive_gcp'
& 'C:\Users\Brandon\Documents\Phase 1\.venv\Scripts\python.exe' Radar_POC\tmd_radar_collect.py report
```

The mirror goes to `TMD_Radar_Archive_gcp`, kept apart from the laptop's own archive. The sync
fetches only files the mirror lacks and is safe to re-run. `report` then shows the VM's
coverage, gaps and lag.

## Day to day

- **New collector version:** copy it and re-run setup. The archive and cron line are kept.
  ```powershell
  scp -i "$HOME\.ssh\tmd_radar_vm" Radar_POC\tmd_radar_collect.py "brandon@${ip}:~/"
  ssh -i "$HOME\.ssh\tmd_radar_vm" "brandon@$ip" "bash ~/setup_vm.sh"
  ```
- **IP changed:** the address is ephemeral, so it changes if the VM is stopped and started
  (not on a reboot). Re-run the `describe` command from step 3.
- **Pause:** `gcloud compute instances stop tmd-radar --zone=us-west1-b` in Cloud Shell. A
  stopped VM releases the address, so it costs nothing, and the disk stays free.
- **Finished:** sync one last time, then run
  `gcloud compute instances delete tmd-radar --zone=us-west1-b`. Deleting the VM deletes its
  disk, archive included.
