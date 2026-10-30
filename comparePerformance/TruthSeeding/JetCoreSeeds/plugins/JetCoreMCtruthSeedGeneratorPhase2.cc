// -*- C++ -*-
//
// Package:    Validation/RecoTrack (or TruthSeeding/JetCoreSeeds)
// Class:      JetCoreMCtruthSeedGeneratorPhase2
//
/**\class JetCoreMCtruthSeedGeneratorPhase2 JetCoreMCtruthSeedGeneratorPhase2.cc

 Description: MC-truth ("perfect") seeds for the JetCore iteration, Phase-2 Inner Tracker version of
              Validation/RecoTrack/plugins/JetCoreMCtruthSeedGenerator.cc.

 What is kept from the Phase-1 module
   * jets ("cores") above ptMin define the regions; SimTracks in a deltaR cone around the jet axis are seeded
   * one target layer for the seeds (default: barrel layer 2, as in DeepCore)
   * hit-less TrajectorySeeds, alongMomentum, with configurable errors (default = DeepCore NN residual widths)
   * a reco::Track per seed for quick inspection
   * a duplicate-seed check based on a packed seed id

 What changes for Phase-2
   * pixel geometry is read from the geometry (PixelTopology pitch, module bounds) instead of hard-coding
     the Phase-1 100x150 um pitch and the 3.35 cm module half-length
   * candidate modules are taken from TrackerGeometry (all Inner-Tracker modules), not from modules that
     happen to have clusters, so no cluster/CPE input is needed
   * forward jets: if the jet axis misses the barrel target layer, the target becomes an endcap disk
     (TFPX/TEPX, PixelEndcap subdet) on the jet's z side
   * seed position/direction (mode "seedOnTrueModule", default):
       the SimTrack's own PSimHit on the target layer/disk -> exact module, local position, local momentum
     legacy mode (seedOnTrueModule = False):
       straight-line extrapolation of the production momentum onto the module crossed by the jet axis,
       as in the Phase-1 code, but the crossing point must lie inside the module bounds
   * true charge of the SimTrack (Phase-1 code always used +1)
   * SimTrack selection: charged, pT > simTrackPtMin, produced inside the target layer, signal only (optional),
     PSimHit/SimTrack match requires the same particle type (rejects hits of un-saved secondaries)
   * one seed per SimTrack per event, even if it is inside several jet cones
   * seed-id packing: unsigned, offset bins with Phase-2 granularity (pixel row/column of the seed module from
     the PixelTopology, plus eta/phi bins) and the module DetId, so negative coordinates and different modules
     can no longer collide (the Phase-1 packing left-shifted signed values and overlapped fields)

 Outputs
   TrajectorySeedCollection
   reco::TrackCollection                 (one per seed: seed position and momentum)
   std::vector<int>  "simTrackIndex"     (index of the SimTrack in the input collection, parallel to the seeds)
*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"

class JetCoreMCtruthSeedGeneratorPhase2 : public edm::one::EDProducer<> {
public:
  explicit JetCoreMCtruthSeedGeneratorPhase2(const edm::ParameterSet&);
  ~JetCoreMCtruthSeedGeneratorPhase2() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ---- target surface: barrel layer or endcap disk (one side) ----
  struct Target {
    int subdet = 0;          // PixelSubdetector::PixelBarrel or PixelEndcap
    unsigned int layer = 0;  // barrel layer or endcap disk number
    unsigned int side = 0;   // endcap only: TrackerTopology::side (1 = -z, 2 = +z); 0 = any
  };
  bool onTarget(const DetId& id, const Target& t) const;
  void buildModuleCache(const TrackerGeometry&, const TrackerTopology&);

  // straight line from point along direction, intersected with the module plane
  static std::pair<bool, GlobalPoint> findIntersection(const GlobalVector& dir, const GlobalPoint& from, const GeomDet* det);

  // module of the target layer crossed by the jet axis (closest to the module centre, inside the bounds)
  const PixelGeomDetUnit* detectorSelector(const Target&, const GlobalVector& dir, const GlobalPoint& vtx) const;

  // target for a given jet: barrel layer if crossed, else endcap disk on the jet's z side
  bool jetTarget(const GlobalVector& jetDir, const GlobalPoint& vtx, Target& t, const PixelGeomDetUnit*& det) const;

  // duplicate-seed key with Phase-2 granularity
  using SeedKey = std::tuple<uint32_t, uint64_t>;  // (DetId, packed bins)
  SeedKey seedKey(const PixelGeomDetUnit* det, const LocalPoint& lp, double eta, double phi) const;

  static uint64_t trackKey(uint32_t eventRaw, unsigned int trackId) { return (uint64_t(eventRaw) << 32) | trackId; }

  // ---- tokens ----
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  edm::ESWatcher<TrackerDigiGeometryRecord> geomWatcher_;

  const edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
  const edm::EDGetTokenT<edm::View<reco::Candidate>> coresToken_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  const edm::EDGetTokenT<edm::SimVertexContainer> simVerticesToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;

  // ---- configuration ----
  const double ptMin_;
  const double absEtaMax_;
  const double deltaR_;
  const double simTrackPtMin_;
  const bool signalOnly_;
  const bool requireProducedInside_;
  const unsigned int barrelLayer_;
  const unsigned int endcapDisk_;
  const bool useEndcap_;
  const bool seedOnTrueModule_;
  const bool removeDuplicates_;
  const double boundsTolerance_;  // cm, legacy mode: how far outside the module the extrapolation may land
  const double etaBin_, phiBin_;  // seed-id granularity
  const double errQoverP_, errDxDz_, errDyDz_, errX_, errY_;

  // ---- cached geometry (rebuilt when the geometry changes) ----
  std::vector<const PixelGeomDetUnit*> pixelDets_;
  const TrackerTopology* tTopo_ = nullptr;
};

// ===========================================================================================================

JetCoreMCtruthSeedGeneratorPhase2::JetCoreMCtruthSeedGeneratorPhase2(const edm::ParameterSet& cfg)
    : geomToken_(esConsumes()),
      topoToken_(esConsumes()),
      verticesToken_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
      coresToken_(consumes<edm::View<reco::Candidate>>(cfg.getParameter<edm::InputTag>("cores"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(cfg.getParameter<edm::InputTag>("simTracks"))),
      simVerticesToken_(consumes<edm::SimVertexContainer>(cfg.getParameter<edm::InputTag>("simVertex"))),
      ptMin_(cfg.getParameter<double>("ptMin")),
      absEtaMax_(cfg.getParameter<double>("absEtaMax")),
      deltaR_(cfg.getParameter<double>("deltaR")),
      simTrackPtMin_(cfg.getParameter<double>("simTrackPtMin")),
      signalOnly_(cfg.getParameter<bool>("signalOnly")),
      requireProducedInside_(cfg.getParameter<bool>("requireProducedInside")),
      barrelLayer_(cfg.getParameter<unsigned int>("barrelLayer")),
      endcapDisk_(cfg.getParameter<unsigned int>("endcapDisk")),
      useEndcap_(cfg.getParameter<bool>("useEndcap")),
      seedOnTrueModule_(cfg.getParameter<bool>("seedOnTrueModule")),
      removeDuplicates_(cfg.getParameter<bool>("removeDuplicates")),
      boundsTolerance_(cfg.getParameter<double>("boundsTolerance")),
      etaBin_(cfg.getParameter<double>("seedIdEtaBin")),
      phiBin_(cfg.getParameter<double>("seedIdPhiBin")),
      errQoverP_(cfg.getParameter<double>("errorQoverP")),
      errDxDz_(cfg.getParameter<double>("errorDxDz")),
      errDyDz_(cfg.getParameter<double>("errorDyDz")),
      errX_(cfg.getParameter<double>("errorX")),
      errY_(cfg.getParameter<double>("errorY")) {
  for (const auto& tag : cfg.getParameter<std::vector<edm::InputTag>>("simHits"))
    simHitTokens_.push_back(consumes<edm::PSimHitContainer>(tag));
  produces<TrajectorySeedCollection>();
  produces<reco::TrackCollection>();
  produces<std::vector<int>>("simTrackIndex");
}

// ------------------------------------------------------------------------------------------------------------
void JetCoreMCtruthSeedGeneratorPhase2::buildModuleCache(const TrackerGeometry& geom, const TrackerTopology&) {
  pixelDets_.clear();
  for (const auto* du : geom.detUnits()) {
    const DetId id = du->geographicalId();
    if (id.det() != DetId::Tracker)
      continue;
    if (id.subdetId() != PixelSubdetector::PixelBarrel && id.subdetId() != PixelSubdetector::PixelEndcap)
      continue;
    if (const auto* pdu = dynamic_cast<const PixelGeomDetUnit*>(du))
      pixelDets_.push_back(pdu);
  }
  edm::LogInfo("PerfectSeederPhase2") << "cached " << pixelDets_.size() << " Inner Tracker pixel modules";
}

bool JetCoreMCtruthSeedGeneratorPhase2::onTarget(const DetId& id, const Target& t) const {
  if (int(id.subdetId()) != t.subdet)
    return false;
  if (tTopo_->layer(id) != t.layer)  // pxbLayer for the barrel, pxfDisk for the endcap
    return false;
  return t.subdet == PixelSubdetector::PixelBarrel || t.side == 0 || tTopo_->side(id) == t.side;
}

std::pair<bool, GlobalPoint> JetCoreMCtruthSeedGeneratorPhase2::findIntersection(const GlobalVector& dir,
                                                                                    const GlobalPoint& from,
                                                                                    const GeomDet* det) {
  StraightLinePlaneCrossing line(Basic3DVector<float>(from.x(), from.y(), from.z()),
                                 Basic3DVector<float>(dir.x(), dir.y(), dir.z()));
  // only crossings in the forward direction (s > 0) of the line
  const auto s = line.pathLength(det->specificSurface());
  if (!s.first || s.second <= 0.f)
    return {false, GlobalPoint()};
  const auto pos = line.position(s.second);
  return {true, GlobalPoint(pos.x(), pos.y(), pos.z())};
}

const PixelGeomDetUnit* JetCoreMCtruthSeedGeneratorPhase2::detectorSelector(const Target& t,
                                                                          const GlobalVector& dir,
                                                                          const GlobalPoint& vtx) const {
  const PixelGeomDetUnit* best = nullptr;
  double bestDist = std::numeric_limits<double>::max();
  for (const auto* det : pixelDets_) {
    if (!onTarget(det->geographicalId(), t))
      continue;
    const auto inter = findIntersection(dir, vtx, det);
    if (!inter.first)
      continue;
    const LocalPoint lp = det->surface().toLocal(inter.second);
    // module bounds from the geometry (replaces the Phase-1 |y| < 3.35 cm)
    if (!det->surface().bounds().inside(Local3DPoint(lp.x(), lp.y(), 0.f)))
      continue;
    const double d = std::hypot(lp.x(), lp.y());  // distance from the module centre
    if (d < bestDist) {
      bestDist = d;
      best = det;
    }
  }
  return best;
}

bool JetCoreMCtruthSeedGeneratorPhase2::jetTarget(const GlobalVector& jetDir,
                                                 const GlobalPoint& vtx,
                                                 Target& t,
                                                 const PixelGeomDetUnit*& det) const {
  t = Target{PixelSubdetector::PixelBarrel, barrelLayer_, 0};
  det = detectorSelector(t, jetDir, vtx);
  if (det)
    return true;
  if (!useEndcap_)
    return false;
  t = Target{PixelSubdetector::PixelEndcap, endcapDisk_, jetDir.z() > 0 ? 2u : 1u};
  det = detectorSelector(t, jetDir, vtx);
  return det != nullptr;
}

// Phase-2 seed id. The Phase-1 version was
//   (x*200 << 0) + (y*200 << 16) + (eta*400 << 32) + (phi*400 << 48)
// with signed values (negative x/y/eta/phi bleed into the other fields) and 50 um x/y bins regardless of pitch.
// Here: pixel row/column from the module's own PixelTopology (Phase-2 25x100 um or 50x50 um, big pixels handled
// by the topology), non-negative offset bins for eta/phi, and the DetId kept separately in the key.
JetCoreMCtruthSeedGeneratorPhase2::SeedKey JetCoreMCtruthSeedGeneratorPhase2::seedKey(const PixelGeomDetUnit* det,
                                                                                    const LocalPoint& lp,
                                                                                    double eta,
                                                                                    double phi) const {
  const auto& topo = det->specificTopology();
  const auto px = topo.pixel(lp);  // (row, column) as floats
  const uint64_t row = uint64_t(std::clamp(int(px.first), 0, (1 << 14) - 1));     // 14 bits (Phase-2 max ~1400 rows)
  const uint64_t col = uint64_t(std::clamp(int(px.second), 0, (1 << 12) - 1));    // 12 bits (Phase-2 max ~440 cols)
  const uint64_t ieta = uint64_t(std::clamp(int((eta + 6.) / etaBin_), 0, (1 << 19) - 1));     // 19 bits
  const uint64_t iphi = uint64_t(std::clamp(int((phi + M_PI) / phiBin_), 0, (1 << 19) - 1));   // 19 bits
  const uint64_t packed = row | (col << 14) | (ieta << 26) | (iphi << 45);                    // 64 bits used
  return {det->geographicalId().rawId(), packed};
}

// ------------------------------------------------------------------------------------------------------------
void JetCoreMCtruthSeedGeneratorPhase2::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto seeds = std::make_unique<TrajectorySeedCollection>();
  auto tracks = std::make_unique<reco::TrackCollection>();
  auto simIdx = std::make_unique<std::vector<int>>();

  const auto& geom = iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
  if (geomWatcher_.check(iSetup) || pixelDets_.empty())
    buildModuleCache(geom, *tTopo_);

  const auto& vertices = iEvent.get(verticesToken_);
  const auto& cores = iEvent.get(coresToken_);
  const auto& simTracks = iEvent.get(simTracksToken_);
  const auto& simVertices = iEvent.get(simVerticesToken_);

  // jet vertex: leading reco PV (as in the Phase-1 module), protected against an empty collection
  GlobalPoint pv(0, 0, 0);
  if (!vertices.empty())
    pv = GlobalPoint(vertices[0].x(), vertices[0].y(), vertices[0].z());

  // SimTrack lookup by (EncodedEventId, trackId)
  std::unordered_map<uint64_t, size_t> trackIndex;
  trackIndex.reserve(simTracks.size());
  for (size_t i = 0; i < simTracks.size(); ++i)
    trackIndex.emplace(trackKey(simTracks[i].eventId().rawId(), simTracks[i].trackId()), i);

  // PSimHits on Inner-Tracker modules, per SimTrack, own particle only, time ordered
  std::unordered_map<size_t, std::vector<const PSimHit*>> hitsByTrack;
  std::vector<edm::Handle<edm::PSimHitContainer>> hitHandles(simHitTokens_.size());
  for (size_t t = 0; t < simHitTokens_.size(); ++t) {
    iEvent.getByToken(simHitTokens_[t], hitHandles[t]);
    for (const auto& sh : *hitHandles[t]) {
      const auto it = trackIndex.find(trackKey(sh.eventId().rawId(), sh.trackId()));
      if (it == trackIndex.end() || sh.particleType() != simTracks[it->second].type())
        continue;
      hitsByTrack[it->second].push_back(&sh);
    }
  }
  for (auto& [idx, v] : hitsByTrack)
    std::sort(v.begin(), v.end(), [](const PSimHit* a, const PSimHit* b) { return a->tof() < b->tof(); });

  std::unordered_set<size_t> seededTracks;  // one seed per SimTrack per event
  std::set<SeedKey> ids;                    // duplicate check on the packed seed id

  for (const auto& jet : cores) {
    if (jet.pt() <= ptMin_ || std::abs(jet.eta()) >= absEtaMax_)
      continue;
    const GlobalVector jetDir(jet.px(), jet.py(), jet.pz());

    Target target;
    const PixelGeomDetUnit* jetDet = nullptr;
    if (!jetTarget(jetDir, pv, target, jetDet)) {
      LogDebug("PerfectSeederPhase2") << "jet pt " << jet.pt() << " eta " << jet.eta() << ": no target module";
      continue;
    }
    // radius / |z| of the target surface, for the "produced inside" requirement
    const double targetR = jetDet->surface().position().perp();
    const double targetZ = std::abs(jetDet->surface().position().z());

    unsigned int nCone = 0, nSeeds = 0;
    for (size_t i = 0; i < simTracks.size(); ++i) {
      const SimTrack& st = simTracks[i];
      const int q = int(std::lround(st.charge()));
      if (q == 0 || st.momentum().pt() < simTrackPtMin_)
        continue;
      if (signalOnly_ && (st.eventId().bunchCrossing() != 0 || st.eventId().event() != 0))
        continue;
      if (reco::deltaR(jet.eta(), jet.phi(), st.momentum().eta(), st.momentum().phi()) >= deltaR_)
        continue;
      if (st.noVertex() || st.vertIndex() >= int(simVertices.size()))
        continue;
      const auto& vpos = simVertices[st.vertIndex()].position();
      if (requireProducedInside_) {
        const bool inside = target.subdet == PixelSubdetector::PixelBarrel ? vpos.Rho() < targetR - 0.5
                                                                           : std::abs(vpos.Z()) < targetZ - 0.5;
        if (!inside)
          continue;
      }
      ++nCone;
      if (seededTracks.count(i))
        continue;

      const PixelGeomDetUnit* det = nullptr;
      LocalPoint lpos;
      LocalVector lmom;

      if (seedOnTrueModule_) {
        // the SimTrack's own hit on the target layer/disk: exact module, position and momentum
        const auto h = hitsByTrack.find(i);
        if (h == hitsByTrack.end())
          continue;
        const PSimHit* hit = nullptr;
        for (const auto* sh : h->second)
          if (onTarget(DetId(sh->detUnitId()), target)) {
            hit = sh;
            break;
          }
        if (!hit)
          continue;
        det = dynamic_cast<const PixelGeomDetUnit*>(geom.idToDetUnit(DetId(hit->detUnitId())));
        if (!det)
          continue;
        lpos = LocalPoint(hit->localPosition().x(), hit->localPosition().y(), 0.f);
        lmom = hit->momentumAtEntry();
      } else {
        // legacy: straight-line extrapolation of the production momentum onto the jet's module
        det = jetDet;
        const GlobalVector p0(st.momentum().x(), st.momentum().y(), st.momentum().z());
        const auto inter = findIntersection(p0, GlobalPoint(vpos.x(), vpos.y(), vpos.z()), det);
        if (!inter.first)
          continue;
        lpos = det->surface().toLocal(inter.second);
        lpos = LocalPoint(lpos.x(), lpos.y(), 0.f);
        const auto& b = det->surface().bounds();
        if (std::abs(lpos.x()) > 0.5 * b.width() + boundsTolerance_ ||
            std::abs(lpos.y()) > 0.5 * b.length() + boundsTolerance_)
          continue;
        lmom = det->surface().toLocal(p0);
      }

      const GlobalVector gmom = det->surface().toGlobal(lmom);
      const GlobalPoint gpos = det->surface().toGlobal(lpos);

      const SeedKey key = seedKey(det, lpos, gmom.eta(), gmom.phi());
      if (ids.count(key)) {
        LogDebug("PerfectSeederPhase2") << "duplicate seed id for SimTrack " << st.trackId();
        if (removeDuplicates_)
          continue;
      }
      ids.insert(key);
      seededTracks.insert(i);

      float em[15] = {0};  // lower-triangular packed covariance (see LocalTrajectoryError)
      em[0] = errQoverP_ * errQoverP_;
      em[2] = errDxDz_ * errDxDz_;
      em[5] = errDyDz_ * errDyDz_;
      em[9] = errX_ * errX_;
      em[14] = errY_ * errY_;

      const LocalTrajectoryParameters par(lpos, lmom, TrackCharge(q));
      seeds->emplace_back(PTrajectoryStateOnDet(par, gmom.perp(), em, det->geographicalId().rawId(), 0),
                          TrajectorySeed::RecHitContainer(),
                          alongMomentum);
      reco::Track::CovarianceMatrix cov;
      tracks->emplace_back(0.,
                           0.,
                           reco::Track::Point(gpos.x(), gpos.y(), gpos.z()),
                           reco::Track::Vector(gmom.x(), gmom.y(), gmom.z()),
                           q,
                           cov);
      simIdx->push_back(int(i));
      ++nSeeds;

      LogDebug("PerfectSeederPhase2") << "seed: SimTrack " << st.trackId() << " pdg " << st.type() << " q " << q
                                      << " pt " << gmom.perp() << " eta " << gmom.eta() << " phi " << gmom.phi()
                                      << " det " << det->geographicalId().rawId() << " subdet "
                                      << det->geographicalId().subdetId() << " layer "
                                      << tTopo_->layer(det->geographicalId());
    }
    edm::LogInfo("PerfectSeederPhase2") << "jet pt " << jet.pt() << " eta " << jet.eta() << ": " << nCone
                                        << " SimTracks in cone, " << nSeeds << " new seeds on "
                                        << (target.subdet == PixelSubdetector::PixelBarrel ? "barrel layer " : "disk ")
                                        << target.layer;
  }

  iEvent.put(std::move(seeds));
  iEvent.put(std::move(tracks));
  iEvent.put(std::move(simIdx), "simTrackIndex");
}

// ------------------------------------------------------------------------------------------------------------
void JetCoreMCtruthSeedGeneratorPhase2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertices", edm::InputTag("firstStepPrimaryVertices"));
  desc.add<edm::InputTag>("cores", edm::InputTag("jetsForCoreTracking"));
  desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("simVertex", edm::InputTag("g4SimHits"));
  desc.add<std::vector<edm::InputTag>>("simHits",
                                       {edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof")});
  desc.add<double>("ptMin", 100.)->setComment("jet pT threshold (Phase-1 default was 300)");
  desc.add<double>("absEtaMax", 4.0);
  desc.add<double>("deltaR", 0.1)->setComment("dR(jet axis, SimTrack production momentum); Phase-1 default 0.3");
  desc.add<double>("simTrackPtMin", 0.9);
  desc.add<bool>("signalOnly", true);
  desc.add<bool>("requireProducedInside", true)->setComment("SimTrack produced inside the target layer/disk");
  desc.add<unsigned int>("barrelLayer", 2)->setComment("TBPX layer used as seeding surface");
  desc.add<unsigned int>("endcapDisk", 1)->setComment("PixelEndcap disk (TFPX/TEPX numbering) for forward jets");
  desc.add<bool>("useEndcap", true)->setComment("use the endcap disk when the jet axis misses the barrel layer");
  desc.add<bool>("seedOnTrueModule", true)
      ->setComment("True: seed state from the SimTrack's own PSimHit on the target layer/disk. "
                   "False: Phase-1-like straight-line extrapolation onto the jet's module");
  desc.add<bool>("removeDuplicates", true)->setComment("skip seeds with an already used seed id");
  desc.add<double>("boundsTolerance", 0.05)->setComment("cm, legacy mode only");
  desc.add<double>("seedIdEtaBin", 0.0025);
  desc.add<double>("seedIdPhiBin", 0.0025);
  // DeepCore NN residual widths (variances 0.15^2, 0.5e-5, 0.5e-5, 2e-5, 2e-5 in the Phase-1 code)
  desc.add<double>("errorQoverP", 0.15);
  desc.add<double>("errorDxDz", std::sqrt(0.5e-5));
  desc.add<double>("errorDyDz", std::sqrt(0.5e-5));
  desc.add<double>("errorX", std::sqrt(2e-5));
  desc.add<double>("errorY", std::sqrt(2e-5));
  descriptions.add("JetCoreMCtruthSeedGeneratorPhase2", desc);
}

DEFINE_FWK_MODULE(JetCoreMCtruthSeedGeneratorPhase2);
