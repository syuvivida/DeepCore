#!/bin/bash
# usage: bash harvest_all.sh <prefix> <eos_outdir> [tags...]
#   e.g. bash harvest_all.sh PU200 /eos/user/s/syu/deepcore_val/PU200
# Harvests every tag, continues after a failure, prints a summary at the end.
PRE=${1:?prefix (e.g. PU200)}; OUT=${2:?eos outdir}; shift 2
TAGS=${@:-nojetcore base base_dcbuild base_rebuild deepcore replace combined combined_dcbuild combined_rebuild deepcore_rebuild}
HERE=$(dirname $(readlink -f $0))
ok=(); bad=()
for t in $TAGS; do
  # optional: wait until no job of this tag is still in the queue
  if [ -f jobs_${PRE}_$t/log/job.log ] && command -v condor_wait >/dev/null; then
    echo "[${PRE}_$t] waiting for jobs to finish ..."; condor_wait jobs_${PRE}_$t/log/job.log >/dev/null 2>&1
  fi
  if bash $HERE/harvest.sh ${PRE}_$t $OUT < /dev/null; then ok+=(${PRE}_$t); else bad+=(${PRE}_$t); fi
done
echo; echo "done:   ${ok[*]}"; echo "failed: ${bad[*]}"
