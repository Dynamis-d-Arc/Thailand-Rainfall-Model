# Hourly predict: Google Cloud Scheduler as the clock

Written 2026-11-10. **Live since 2026-11-10 03:20 UTC**: the job's first firing started run
42748742826 at 03:20:02. The steps below are kept as the record of how it was set up.

## Why

The hourly forecast is computed by GitHub Actions (`.github/workflows/hourly-predict.yml`), but
something has to start it each hour. GitHub's own cron fires in only a minority of hours, so a
Claude routine ("Rainwatch hourly-predict keeper") dispatches the workflow at :25. That routine
is an AI agent and sometimes goes off script and gets blocked. Between 25 Oct and 10 Nov 2026
the `predictions` branch is missing 13 of 394 hourly files, most of them from the keeper.

A Cloud Scheduler job sends the same dispatch request at :20 every hour, with no agent involved.

- It only starts the run. The model, the workflow and the output are unchanged.
- The keeper stays as a backup. It skips any hour that already has a run, so the two do not
  double up.
- It does not help when GitHub Actions itself is down (2 incidents, 3 lost hours, since
  October). Those hours stay lost and are reported as such.
- Cost: none. Three scheduler jobs are free per billing account.

## 1. Create the GitHub token (you, signed in as Dynamis-d-Arc)

The repo belongs to the `Dynamis-d-Arc` user account, and a fine-grained token only reaches
repos its own account owns. A token made in the BRANDON-VEKIN account will get a 404.

1. GitHub → profile picture → Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token.
2. Name: `rainwatch-scheduler`. Expiration: custom, 31 May 2027.
3. Repository access: Only select repositories → `Thailand-Rainfall-Model`.
4. Permissions → Repository permissions → Actions: **Read and write**. Nothing else.
5. Generate, and copy the token. It is shown once.

## 2. Open Cloud Shell

Go to https://console.cloud.google.com, pick the project `tmd-radar-poc`, and click the
terminal icon (Activate Cloud Shell) at the top right. Nothing needs installing.

## 3. Create the job

Paste these one at a time. The second asks for the token and hides what you paste, so it does
not land in the shell history.

```bash
gcloud config set project tmd-radar-poc && gcloud services enable cloudscheduler.googleapis.com
```

```bash
read -s -p "GitHub token: " GH_TOKEN; echo
```

```bash
gcloud scheduler jobs create http rainwatch-hourly-dispatch \
  --location=us-west1 \
  --schedule="20 * * * *" --time-zone="Etc/UTC" \
  --uri="https://api.github.com/repos/Dynamis-d-Arc/Thailand-Rainfall-Model/actions/workflows/hourly-predict.yml/dispatches" \
  --http-method=POST \
  --headers="Authorization=Bearer ${GH_TOKEN},Accept=application/vnd.github+json,Content-Type=application/json,X-GitHub-Api-Version=2022-11-28" \
  --message-body='{"ref":"main"}' \
  --attempt-deadline=30s --max-retry-attempts=3 --min-backoff=30s --max-backoff=120s
```

Minute 20 is deliberate. It is before the keeper's :25 check, and a run started then finishes
well inside the hour. A run that crosses the hour boundary forecasts the wrong hour.

## 4. Test it

Only between :05 and :45 past the hour, for the same reason:

```bash
gcloud scheduler jobs run rainwatch-hourly-dispatch --location=us-west1
```

```bash
gcloud scheduler jobs describe rainwatch-hourly-dispatch --location=us-west1 --format="yaml(state,lastAttemptTime,status)"
```

- Good: `state: ENABLED`, a fresh `lastAttemptTime`, and `status` empty. A new run appears at
  https://github.com/Dynamis-d-Arc/Thailand-Rainfall-Model/actions within a few seconds.
- `status` with code 5 (404) or 7 (401/403): the token is wrong, from the wrong account, or
  lacks Actions write. Code 3 (415/422): a header or the body was mangled when pasting.

The test adds one extra run for that hour, which rewrites the same hour's file. That is harmless.

## 5. Afterwards

- From the next hour, runs should start at :20 UTC. Check with
  `gh run list --workflow hourly-predict.yml --limit 30`.
- The token is stored readable in the job's settings, so anyone with access to the
  `tmd-radar-poc` project can see it. It can only start and read Actions runs on this one repo.
- It expires on the date set in step 1. To replace it, repeat step 1 and run
  `gcloud scheduler jobs update http rainwatch-hourly-dispatch --location=us-west1 --update-headers="Authorization=Bearer NEW_TOKEN"`.
- To stop: `gcloud scheduler jobs pause rainwatch-hourly-dispatch --location=us-west1`, and
  delete the token on GitHub.
