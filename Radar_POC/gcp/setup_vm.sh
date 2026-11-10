#!/usr/bin/env bash
# Run on the Debian 12 VM, with tmd_radar_collect.py copied next to this script.
# Installs the collector, checks it, does one live run and adds the 5-minute cron job.
# Safe to re-run after copying a newer tmd_radar_collect.py.
set -euo pipefail

APP="$HOME/tmd_radar"
ARCHIVE="$HOME/tmd_radar_archive"
HERE="$(cd "$(dirname "$0")" && pwd)"

sudo apt-get update -q
sudo apt-get install -y -q python3-numpy python3-scipy python3-pil cron

mkdir -p "$APP" "$ARCHIVE"
cp "$HERE/tmd_radar_collect.py" "$APP/"
python3 "$APP/tmd_radar_collect.py" selftest
TMD_RADAR_ARCHIVE="$ARCHIVE" python3 "$APP/tmd_radar_collect.py" run

# flock: cron, unlike Task Scheduler, would start a second run while a slow one is still going
LINE="*/5 * * * * TMD_RADAR_ARCHIVE=$ARCHIVE flock -n /tmp/tmd_radar.lock /usr/bin/python3 $APP/tmd_radar_collect.py run --log-file $ARCHIVE/collector.log"
# Healthchecks.io alert, only once its ping URL has been put in ~/.tmd_radar_healthcheck_url
HC=""
if [ -s "$HOME/.tmd_radar_healthcheck_url" ]; then
  cp "$HERE/healthcheck.sh" "$APP/"
  HC="7,22,37,52 * * * * TMD_RADAR_ARCHIVE=$ARCHIVE bash $APP/healthcheck.sh"
fi
( crontab -l 2>/dev/null | grep -v -e tmd_radar_collect.py -e healthcheck.sh || true
  echo "$LINE"
  if [ -n "$HC" ]; then echo "$HC"; fi ) | crontab -
echo "cron installed:"
crontab -l
