#!/usr/bin/env bash
# Cron, every 15 min: tell Healthchecks.io whether a fresh radar frame is archived.
# Success ping while the newest NetCDF is under an hour old, /fail ping (with the reason) once
# it isn't; if the VM or cron dies the pings stop, and Healthchecks alerts on the silence.
# The ping URL lives in ~/.tmd_radar_healthcheck_url, never in git.
URL="$(cat "$HOME/.tmd_radar_healthcheck_url")"
if msg="$(python3 "$HOME/tmd_radar/tmd_radar_collect.py" health 2>&1)"; then
  curl -fsS -m 10 --retry 3 --data-raw "$msg" "$URL" > /dev/null
else
  curl -fsS -m 10 --retry 3 --data-raw "$msg" "$URL/fail" > /dev/null
fi
