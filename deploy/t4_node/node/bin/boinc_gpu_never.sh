#!/bin/bash
# After the BOINC client starts: wait for its RPC port, then switch GPU work OFF (the governor turns it on when the node is idle).
D=/mnt/disks/disk-socrateai-local-1/boinc/node/client
export LD_LIBRARY_PATH=/mnt/disks/disk-socrateai-local-1/boinc/root/usr/lib/x86_64-linux-gnu
CMD=/mnt/disks/disk-socrateai-local-1/boinc/root/usr/bin/boinccmd
for i in $(seq 1 90); do
  if [ -f "$D/gui_rpc_auth.cfg" ] && "$CMD" --host 127.0.0.1:31420 --passwd "$(cat $D/gui_rpc_auth.cfg)" --get_state >/dev/null 2>&1; then
    "$CMD" --host 127.0.0.1:31420 --passwd "$(cat $D/gui_rpc_auth.cfg)" --set_gpu_mode never 0
    exit 0
  fi
  sleep 1
done
exit 0
