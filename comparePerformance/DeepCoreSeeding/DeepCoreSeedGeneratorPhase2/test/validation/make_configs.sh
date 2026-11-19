#!/bin/bash
# Make the step3 validation configs (DQM output only, no event content) for each setup.
# usage: bash make_configs.sh [nThreads]      (run in a directory of your choice, after cmsenv)
NT=${1:-8}
C=DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/customise_DeepCorePhase2
VAL=$C.customiseJetCoreValidation
DCB="process.jetCoreRegionalStepTrajectoryBuilder.keepOriginalIfRebuildFails = True; process.jetCoreRegionalStepTrajectoryBuilder.requireSeedHitsInRebuild = False; process.jetCoreRegionalStepTrajectoryBuilder.lockHits = False; process.jetCoreRegionalStepTrackCandidates.TrajectoryCleaner = 'jetCoreRegionalStepDeepCoreTrajectoryCleaner'"

REB="process.jetCoreRegionalStepTrajectoryBuilder.keepOriginalIfRebuildFails = True; process.jetCoreRegionalStepTrajectoryBuilder.requireSeedHitsInRebuild = False"

# two chains as in "deepcore" (DeepCore seeds and pixel-pair seeds built separately, merged at track level), but both
# chains use the standard jet-core builder with the two rebuild settings ($REB) and the default trajectory cleaner,
# i.e. the same build settings as combined_rebuild. The standard builder also brings its trajectory filter.
SEP="process.jetCoreRegionalStepBarrelTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'jetCoreRegionalStepTrajectoryBuilder'; process.jetCoreRegionalStepEndcapTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'jetCoreRegionalStepTrajectoryBuilder'; process.jetCoreRegionalStepBarrelTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'; $REB"

make() {  # name, customise list, [customise_commands], [procModifiers]
  local extra=()
  [ -n "$3" ] && extra=(--customise_commands "$3")
  local pm=${4:-splitClustersInPhase2Pixel,jetCoreInPhase2}
  cmsDriver.py step3 --conditions auto:phase2_realistic_T35 --era Phase2C26I13M9 --geometry ExtendedRun4D128 \
    --datatier DQMIO --eventcontent DQM \
    -s RAW2DIGI,RECO,RECOSIM,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
    --procModifiers $pm \
    --customise "$2" "${extra[@]}" \
    --filein file:dummy.root --fileout file:step3_$1.root \
    --nStreams 1 --nThreads $NT -n -1 --python_filename step3_$1.py --no_exec > /dev/null && echo "made step3_$1.py"
}
# reference without any jet-core step (Phase-2 default + pixel cluster splitting)
make nojetcore        "$VAL" "" splitClustersInPhase2Pixel
make base             "$VAL"
make base_dcbuild     "$VAL" "$DCB"
make deepcore         "$C.customise,$VAL"
make replace          "$C.customiseReplaceBarrel,$VAL"
make combined         "$C.customiseCombined,$VAL"
make combined_dcbuild "$C.customiseCombined,$VAL" "$DCB"
# reduced settings: only the two rebuild settings that seeds without hits need
make base_rebuild     "$VAL" "$REB"
make combined_rebuild "$C.customiseCombined,$VAL" "$REB"
# two separate chains, same build settings as combined_rebuild: isolates one chain vs two chains
make deepcore_rebuild "$C.customise,$VAL" "$SEP"

# DeepCore seed duplicate removal (event-wide, highest probability kept): two settings, on the DeepCore-only
# layout (replace) to isolate the effect, and the medium one on the full layout
DUP1="process.jetCoreRegionalStepSeedsBarrel.duplicateMaxDist = 0.005; process.jetCoreRegionalStepSeedsBarrel.duplicateMaxDEta = 0.001; process.jetCoreRegionalStepSeedsBarrel.duplicateMaxDPhi = 0.001"
DUP2="process.jetCoreRegionalStepSeedsBarrel.duplicateMaxDist = 0.01; process.jetCoreRegionalStepSeedsBarrel.duplicateMaxDEta = 0.002; process.jetCoreRegionalStepSeedsBarrel.duplicateMaxDPhi = 0.002"
make replace_dup1     "$C.customiseReplaceBarrel,$VAL" "$DUP1"
make replace_dup2     "$C.customiseReplaceBarrel,$VAL" "$DUP2"
make deepcore_dup2    "$C.customise,$VAL" "$DUP2"

# ---- jet-core building/selection studies (Nov 2026) ----
# 45% trajectory cleaner (DeepCore cleaner) on the pixel-pair candidates, and a layer requirement in the jet-core
# selector used by earlyGeneralTracks (in the two-chain layout it acts on the merged jetCoreRegionalStepTracks)
C45="process.jetCoreRegionalStepTrackCandidates.TrajectoryCleaner = 'jetCoreRegionalStepDeepCoreTrajectoryCleaner'"
C45E="process.jetCoreRegionalStepEndcapTrackCandidates.TrajectoryCleaner = 'jetCoreRegionalStepDeepCoreTrajectoryCleaner'"
lay() { echo "[setattr(ps, 'minNumberLayers', $1) or setattr(ps, 'minNumber3DLayers', 4) for ps in process.jetCoreRegionalStepSelector.trackSelectors]"; }
make base_cleaner45        "$VAL" "$REB; $C45"
make base_cleaner45_lay7   "$VAL" "$REB; $C45; $(lay 7)"
make base_cleaner45_lay9   "$VAL" "$REB; $C45; $(lay 9)"

# what does DeepCore itself add? control: two-chain layout without DeepCore seeds (no jet passes ptMin, no inference)
make deepcore_noDC         "$C.customise,$VAL" "process.jetCoreRegionalStepSeedsBarrel.ptMin = 1e9"
# DeepCore + the better pixel-pair building (+ layer requirement)
make deepcore_c45          "$C.customise,$VAL" "$C45E"
make deepcore_c45_lay7     "$C.customise,$VAL" "$C45E; $(lay 7)"
make deepcore_noDC_c45_lay7 "$C.customise,$VAL" "process.jetCoreRegionalStepSeedsBarrel.ptMin = 1e9; $C45E; $(lay 7)"

# DeepCore seed changes in the generator
SB=process.jetCoreRegionalStepSeedsBarrel
make deepcore_q0           "$C.customise,$VAL" "$SB.seedCharge = 0"
make deepcore_q0_qerr      "$C.customise,$VAL" "$SB.seedCharge = 0; $SB.seedQoverPRelError = 1.0"
# with a hit the seed is subject to the candidate maker's seed cleaner: switch that off for the DeepCore chain
make deepcore_hit          "$C.customise,$VAL" "$SB.attachHit = True; process.jetCoreRegionalStepBarrelTrackCandidates.RedundantSeedCleaner = 'none'"
make deepcore_hit_q0       "$C.customise,$VAL" "$SB.attachHit = True; $SB.seedCharge = 0; process.jetCoreRegionalStepBarrelTrackCandidates.RedundantSeedCleaner = 'none'"
