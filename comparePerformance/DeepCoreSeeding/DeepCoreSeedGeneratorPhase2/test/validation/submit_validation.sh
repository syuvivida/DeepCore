#!/bin/bash
# usage: bash submit_validation.sh <tag> <step3_config.py> <filelist.txt> <eos_outdir> [JobFlavour] [maxEventsPerJob]
#   one job per input file; DQM files go to <eos_outdir>/<tag>/<tag>_<n>.root
TAG=${1:?tag}; CFG=$(readlink -f ${2:?config}); LIST=$(readlink -f ${3:?filelist}); OUT=${4:?eos outdir}
FL=${5:-workday}; MAXEV=${6:--1}
HERE=$(dirname $(readlink -f $0))
case $PWD in /eos/*) echo "Submit from AFS (e.g. ~/val), not from EOS: HTCondor cannot write its logs on EOS"; exit 1;; esac
# grid proxy for reading /store via xrootd, kept on AFS so the schedd can ship it
export X509_USER_PROXY=${X509_USER_PROXY:-$HOME/private/x509up}
if ! voms-proxy-info -file $X509_USER_PROXY -exists -valid 24:00 >/dev/null 2>&1; then
  mkdir -p $HOME/private && voms-proxy-init -voms cms -valid 192:00 -out $X509_USER_PROXY || exit 1
fi
mkdir -p jobs_$TAG/{log,out,err}
awk 'NF{print NR-1, $1}' $LIST > jobs_$TAG/inputs.txt
cat > jobs_$TAG/job.sub <<SUB
executable            = $HERE/run_val_job.sh
arguments             = $CMSSW_BASE $CFG \$(infile) $OUT $TAG \$(jobid) $MAXEV
output                = jobs_$TAG/out/\$(jobid).out
error                 = jobs_$TAG/err/\$(jobid).err
log                   = jobs_$TAG/log/job.log
request_cpus          = 8
MY.WantOS             = "el9"
x509userproxy         = $X509_USER_PROXY
+JobFlavour           = "$FL"
MY.SendCredential     = True
getenv                = False
queue jobid, infile from jobs_$TAG/inputs.txt
SUB
echo "$(wc -l < jobs_$TAG/inputs.txt) jobs for $TAG"
condor_submit jobs_$TAG/job.sub
