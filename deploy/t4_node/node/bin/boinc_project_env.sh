#!/bin/bash
# Run a command inside the real BOINC project directory with the python shim and BOINC's python modules on the path.
S=/mnt/disks/disk-socrateai-local-1/boinc/stack
export PATH=$S/shims:$S/project/hmproj/bin:$PATH
export PYTHONPATH=/mnt/disks/disk-socrateai-local-1/boinc/src/py:$S/project/hmproj/py
cd $S/project/hmproj || exit 1
exec "$@"
