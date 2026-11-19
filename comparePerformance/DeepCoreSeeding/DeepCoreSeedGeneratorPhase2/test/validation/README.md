# MTV validation: baseline vs DeepCore (Phase 2)

All commands on lxplus, after `cmsenv` in CMSSW_20_1_0_pre3 with DeepCoreSeeding built (`scram b`).

1. Input file lists (PU200 and noPU of the same RelVal):
   ```bash
   voms-proxy-init -voms cms
   dasgoclient -query="dataset dataset=/RelValQCD_Pt_1800_2400_14/CMSSW_20_1_0_pre3*/GEN-SIM-DIGI-RAW"
   dasgoclient -query="file dataset=<the PU200 dataset>" > files_PU200.txt
   ```
2. Configs (DQM output only): `bash make_configs.sh` -> step3_{nojetcore,base,base_dcbuild,base_rebuild,deepcore,replace,combined,combined_dcbuild,combined_rebuild,deepcore_rebuild}.py
3. Quick local test: `cmsRun` one config on 2 events (add `process.maxEvents.input=2` and a file name), check it ends with no exception.
4. Submit (one job per input file; same files for every config):
   ```bash
   OUT=/eos/user/s/syu/deepcore_val/PU200
   for t in nojetcore base base_dcbuild base_rebuild deepcore replace combined combined_dcbuild combined_rebuild deepcore_rebuild; do
     bash submit_validation.sh PU200_$t step3_$t.py files_PU200.txt $OUT workday
   done
   ```
5. Harvest all tags (waits for the jobs, continues after failures, prints a summary):
   `bash harvest_all.sh PU200 $OUT` -> DQM_PU200_<tag>.root, logs in harvest_PU200_<tag>.log
6. Compare:
   ```bash
   makeTrackValidationPlots.py --extended -o plots_PU200 DQM_PU200_base.root DQM_PU200_base_dcbuild.root DQM_PU200_deepcore.root DQM_PU200_replace.root DQM_PU200_combined_dcbuild.root
   python3 ../compare_mtv_jetcore.py --folder JetCore --files DQM_PU200_base.root DQM_PU200_base_dcbuild.root DQM_PU200_combined_dcbuild.root --names base base+dcbuild combined+dcbuild --out PU200_jetcore.pdf
   python3 ../compare_mtv_jetcore.py --folder Track --labels generalTracks --files DQM_PU200_base.root DQM_PU200_base_dcbuild.root DQM_PU200_combined_dcbuild.root --names base base+dcbuild combined+dcbuild --out PU200_general.pdf
   ```
