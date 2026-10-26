#!/bin/bash
# boinccmd against the T4 node's BOINC client, e.g.  boinccmd_node.sh --get_tasks   |   boinccmd_node.sh --set_gpu_mode never 0
D=/mnt/disks/disk-socrateai-local-1/boinc/node/client
export LD_LIBRARY_PATH=/mnt/disks/disk-socrateai-local-1/boinc/root/usr/lib/x86_64-linux-gnu
exec /mnt/disks/disk-socrateai-local-1/boinc/root/usr/bin/boinccmd --host 127.0.0.1:31420 --passwd "$(cat "$D/gui_rpc_auth.cfg")" "$@"
