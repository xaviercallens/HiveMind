#!/bin/bash
# Install the T4 node's batch layer + boot tooling under /opt/hivemind/t4 and register the systemd units. Run as root:
#   sudo bash deploy/t4_node/install_node.sh
# Idempotent. It copies a snapshot of the code (the git worktree this runs from may be deleted later), installs and enables the
# node units, and DOES NOT start anything: starting is a separate step (bin/startup.sh) so the owner controls when services move.
#
# "Only via repo" (owner, 2026-10-26): the CONTROL PLANE (governor, monitor, work-generator scripts, helper shell scripts and every unit file) is deployed from
# COMMITTED git content (HEAD), never from the working tree, so an uncommitted edit cannot reach the live node through this installer; a skipped uncommitted
# change is reported. The pipeline sources under src/ are still copied from the working tree (documented exception: the owner asked to leave a contributor's
# uncommitted edits there for later); the installer lists them. A hash manifest of what was installed lets the monitor flag later hand edits (CODE_DRIFT).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OPT=/opt/hivemind/t4
USER_NAME=callensxavier_gmail_com
GIT=(git -C "$REPO" -c "safe.directory=$REPO")

committed_to() {  # atomically write the committed content of a repo-relative path to dst; refuse untracked files without truncating dst
  local rel="$1" dst="$2" mode="${3:-0644}"
  if ! "${GIT[@]}" cat-file -e "HEAD:$rel" 2>/dev/null; then
    echo "WARNING: $rel is untracked in git HEAD; skipping without truncating $dst." >&2
    return 1
  fi
  if ! "${GIT[@]}" diff --quiet HEAD -- "$rel" 2>/dev/null; then
    echo "WARNING: $rel has uncommitted changes; deploying the COMMITTED version (git HEAD). Commit and review the change to deploy it." >&2
  fi
  local tmp="${dst}.tmp.$$"
  "${GIT[@]}" show "HEAD:$rel" > "$tmp"
  chmod "$mode" "$tmp"
  mv -f "$tmp" "$dst"
}

# Make a silent rewrite FAIL LOUDLY: the control-plane files are immutable (chattr +i) between installs. Root can clear the flag on purpose (chattr -i), which then shows
# up in the sudo log; a script that just overwrites the file gets "Operation not permitted". The installer clears the flags before writing and restores them at the end.
PROTECT_FILES=("$OPT/scripts/t4_governor.py" "$OPT/scripts/t4_monitor.py" "$OPT/scripts/t4_workgen.py" "$OPT/expected_governor_args.txt" "$OPT/expected_code_hashes.txt"
               "$OPT/POLICY_READ_ME.txt" /etc/systemd/system/hivemind-t4-governor.service /etc/systemd/system/hivemind-t4-monitor.service /etc/systemd/system/hivemind-t4-workgen.service)
# empty drop-in directories, made immutable: nothing can be added to them, so `mkdir -p …service.d && tee …/override.conf` fails
PROTECT_DIRS=(/etc/systemd/system/hivemind-t4-governor.service.d /etc/systemd/system/hivemind-t4-monitor.service.d /etc/systemd/system/hivemind-t4-workgen.service.d)
unlock() { for f in "${PROTECT_FILES[@]}" "${PROTECT_DIRS[@]}" "$OPT"/bin/*.sh; do if [ -e "$f" ]; then chattr -i "$f" 2>/dev/null || true; fi; done; }
lock() { for f in "${PROTECT_FILES[@]}" "${PROTECT_DIRS[@]}" "$OPT"/bin/*.sh; do if [ -e "$f" ]; then chattr +i "$f" 2>/dev/null || echo "NOTE: could not make $f immutable" >&2; fi; done; }
unlock
trap lock EXIT

install -d -m 0755 -o root -g root "$OPT" "$OPT/bin" "$OPT/scripts" "$OPT/src"
rsync -a --delete --exclude '__pycache__' "$REPO/src/hivemind/" "$OPT/src/hivemind/"
if [ -n "$("${GIT[@]}" status --porcelain -- src 2>/dev/null)" ]; then
  echo "NOTE: src/ is copied from the working tree and has uncommitted changes (deployed as they are):" >&2
  "${GIT[@]}" status --porcelain -- src >&2
fi
for s in t4_governor.py t4_monitor.py t4_workgen.py; do
  committed_to "scripts/$s" "$OPT/scripts/$s" 0755
done
for b in $("${GIT[@]}" ls-tree --name-only HEAD deploy/t4_node/node/bin/ 2>/dev/null || true); do
  committed_to "$b" "$OPT/bin/$(basename "$b")" 0755
done
chown -R root:root "$OPT"

for u in $("${GIT[@]}" ls-tree --name-only HEAD deploy/t4_node/node/ 2>/dev/null || true); do
  case "$(basename "$u")" in
    *.service|*.timer)
      committed_to "$u" "/etc/systemd/system/$(basename "$u")" 0644
      ;;
  esac
done
# the older bee/coder producers' units (mirroring the running processes); installed and enabled so a reboot brings them back
for u in hivemind-live-bee-llama hivemind-live-shield-bee hivemind-live-agent-bee hivemind-live-shield-coder hivemind-live-agent-coder; do
  committed_to "deploy/t4_node/live/$u.service" "/etc/systemd/system/$u.service" 0644
done
# Governance: record (1) the arguments the governor template defines, (2) the hash of every control-plane file installed, so the monitor can flag any
# difference (a changed argument, a drop-in, or an edited script), and warn about drop-in overrides, which silently beat the unit files written above.
"${GIT[@]}" show "HEAD:deploy/t4_node/node/hivemind-t4-governor.service" | grep -m1 '^ExecStart=' | sed -E 's/.*t4_governor\.py //' > "$OPT/expected_governor_args.txt"
( cd "$OPT" && sha256sum scripts/*.py bin/*.sh ) > "$OPT/expected_code_hashes.txt"
chown root:root "$OPT/expected_governor_args.txt" "$OPT/expected_code_hashes.txt"
cat > "$OPT/POLICY_READ_ME.txt" <<'POLICY'
HiveMind T4 node: the control-plane files here are protected ON PURPOSE.

Owner's decision (2026-10-26, window changed 2026-11-04): the governor's policy - resume training only after 1800 s without a request inside the training window
(17:00-22:00 Europe/Paris, when the T4 VM runs), 7200 s otherwise, and at least 4300 MiB of free VRAM - changes ONLY through a reviewed change to the repo
(deploy/t4_node/node/hivemind-t4-governor.service and scripts/t4_governor.py), followed by `sudo bash deploy/t4_node/install_node.sh`.
The "intense-moe" mode (5-10 s) is not wanted. Earlier unauthorised edits (a systemd drop-in at 22:08Z on 25 Oct, code edits at 02:08Z and
03:09Z on 26 Oct) were reverted; they are in /var/log/auth.log and /opt/hivemind/t4/removed-overrides/.

These files are immutable (chattr +i) so that an edit fails loudly instead of silently changing the policy. The installer clears and restores the
flag. If you need a different policy, ask the owner and propose the change in the repo; do not remove the flag by hand.
POLICY
chown root:root "$OPT/POLICY_READ_ME.txt"
for d in "${PROTECT_DIRS[@]}"; do install -d -m 0755 "$d"; done
# real drop-in files only (the protected, empty directories are expected)
overrides="$(find /etc/systemd/system -path '*/hivemind-*.service.d/*' -type f 2>/dev/null || true)"
if [ -n "$overrides" ]; then
  echo "WARNING: systemd drop-in overrides exist and take precedence over the unit files this installer writes:"
  echo "$overrides"
fi
systemctl daemon-reload
# The T4 trains only (owner decision 2026-11-04): its producer heartbeats hivemind-live-agent-bee and -coder are NOT enabled; the RTX 2070 and volunteers serve inference.
for u in hivemind-boinc-db hivemind-boinc-apache hivemind-boinc-daemons hivemind-boinc-client hivemind-t4-governor hivemind-t4-workgen hivemind-t4-monitor \
         hivemind-live-bee-llama hivemind-live-shield-bee hivemind-live-shield-coder; do
  systemctl enable "$u.service" >/dev/null 2>&1 && echo "enabled $u" || echo "FAILED to enable $u"
done
systemctl enable hivemind-boinc-watchdog.timer >/dev/null 2>&1 && echo "enabled hivemind-boinc-watchdog.timer" || echo "FAILED to enable hivemind-boinc-watchdog.timer"
lock
echo "installed to $OPT from committed content (control-plane files immutable); nothing was started"
