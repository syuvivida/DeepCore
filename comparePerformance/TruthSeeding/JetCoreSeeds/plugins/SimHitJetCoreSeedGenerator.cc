// -*- C++ -*-
//
// Package:    TruthSeeding/JetCoreSeeds
// Class:      SimHitJetCoreSeedGenerator
//
/**\class SimHitJetCoreSeedGenerator SimHitJetCoreSeedGenerator.cc TruthSeeding/JetCoreSeeds/plugins/SimHitJetCoreSeedGenerator.cc

 Description: "perfect" (MC-truth) TrajectorySeeds for charged particles inside the
              core of high-pT jets, built directly from Geant4 PSimHits.

 Implementation:
   Starting point: RecoTracker/TkSeedGenerator/plugins/DeepCoreSeedGenerator.cc and
   Validation/RecoTrack/plugins/JetCoreMCtruthSeedGenerator.cc. Differences w.r.t. the latter:

   * The seed state is taken from the SimHit itself (local position at the module mid-plane and
     local momentum at entry), on the module that was actually crossed, instead of a straight-line
     extrapolation of the SimTrack production momentum onto one module chosen along the jet axis.
     The state therefore includes the true curvature, the true module, and the true charge
     (the old code always used charge +1).
   * Hits produced by un-saved secondaries (which inherit the parent's trackId) are rejected by
     requiring PSimHit::particleType() == SimTrack::type().
   * One seed per SimTrack, even if the track is inside several jet cones.
   * Configurable seed layer ("innermost", "BPix2", "FPix1", ...), errors, and optional
     reproducible Gaussian smearing of the seed parameters (consistent with the errors).
   * global::EDProducer (thread-safe), no reconstructed clusters needed.

 Outputs:
   TrajectorySeedCollection       : hitless seeds, alongMomentum (as DeepCore produces)
   reco::TrackCollection          : one "track" per seed (seed point + momentum), for quick checks
   std::vector<int>  "simTrackIndex": index of the generating SimTrack in the input SimTrack
                                      collection, parallel to the seed collection
*/

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

class SimHitJetCoreSeedGenerator : public edm::global::EDProducer<> {
public:
  explicit SimHitJetCoreSeedGenerator(const edm::ParameterSet&);
  ~SimHitJetCoreSeedGenerator() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // (event, trackId) key, so SimTracks and SimHits from different EncodedEventIds never mix
  static uint64_t trackKey(uint32_t eventRawId, unsigned int trackId) {
    return (uint64_t(eventRawId) << 32) | uint64_t(trackId);
  }

  // subdetector/layer requested for the seed hit; subdet == 0 means "innermost hit"
  struct SeedLayer {
    int subdet = 0;
    unsigned int layer = 0;
  };
  static SeedLayer parseSeedLayer(const std::string&);
  bool isPixel(const DetId& id) const { return id.subdetId() == 1 || id.subdetId() == 2; }

  // ---------------- tokens ----------------
  const edm::EDGetTokenT<edm::View<reco::Candidate>> coresToken_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  const edm::EDGetTokenT<edm::SimVertexContainer> simVerticesToken_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  // ---------------- jet / track selection ----------------
  const double jetPtMin_;
  const double jetAbsEtaMax_;
  const double deltaR_;        // dR(jet, track momentum at seed hit)
  const double hitDeltaR_;     // dR(jet, seed-hit position seen from sim PV); <=0 disables
  const double simTrackPtMin_;
  const double maxProdRadius_;  // cm; <0 disables
  const bool signalOnly_;       // only SimTracks from the signal (bunch crossing 0, event 0)
  const unsigned int minPixelLayers_;

  // ---------------- seed definition ----------------
  const SeedLayer seedLayer_;
  const bool fallbackToInnermost_;
  const double minHitMomentumFraction_;  // reject hits where |p_hit| < frac * |p_sim| (hard brems/interactions)

  // ---------------- errors and smearing ----------------
  const double errQoverP_;     // absolute sigma(q/p) [1/GeV]
  const double errQoverPRel_;  // relative sigma(q/p), added in quadrature; <=0 disables
  const double errDxDz_;
  const double errDyDz_;
  const double errX_;  // cm
  const double errY_;  // cm
  const bool smear_;
  const unsigned int randomSeed_;
};

SimHitJetCoreSeedGenerator::SeedLayer SimHitJetCoreSeedGenerator::parseSeedLayer(const std::string& s) {
  SeedLayer out;
  if (s == "innermost" || s.empty())
    return out;
  static const std::vector<std::pair<std::string, int>> prefixes = {
      {"BPix", 1}, {"FPix", 2}, {"TIB", 3}, {"TID", 4}, {"TOB", 5}, {"TEC", 6}};
  for (const auto& [name, subdet] : prefixes) {
    if (s.rfind(name, 0) == 0 && s.size() > name.size()) {
      try {
        out.subdet = subdet;
        out.layer = std::stoul(s.substr(name.size()));
        return out;
      } catch (...) {
        break;
      }
    }
  }
  throw cms::Exception("Configuration") << "SimHitJetCoreSeedGenerator: invalid seedLayer '" << s
                                        << "'. Use 'innermost' or e.g. 'BPix2', 'FPix1', 'TIB1'.";
}

SimHitJetCoreSeedGenerator::SimHitJetCoreSeedGenerator(const edm::ParameterSet& iConfig)
    : coresToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("cores"))),
      simTracksToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTracks"))),
      simVerticesToken_(consumes<edm::SimVertexContainer>(iConfig.getParameter<edm::InputTag>("simVertices"))),
      geomToken_(esConsumes()),
      topoToken_(esConsumes()),
      jetPtMin_(iConfig.getParameter<double>("jetPtMin")),
      jetAbsEtaMax_(iConfig.getParameter<double>("jetAbsEtaMax")),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      hitDeltaR_(iConfig.getParameter<double>("hitDeltaR")),
      simTrackPtMin_(iConfig.getParameter<double>("simTrackPtMin")),
      maxProdRadius_(iConfig.getParameter<double>("maxProductionRadius")),
      signalOnly_(iConfig.getParameter<bool>("signalOnly")),
      minPixelLayers_(iConfig.getParameter<unsigned int>("minPixelLayers")),
      seedLayer_(parseSeedLayer(iConfig.getParameter<std::string>("seedLayer"))),
      fallbackToInnermost_(iConfig.getParameter<bool>("fallbackToInnermost")),
      minHitMomentumFraction_(iConfig.getParameter<double>("minHitMomentumFraction")),
      errQoverP_(iConfig.getParameter<double>("errorQoverP")),
      errQoverPRel_(iConfig.getParameter<double>("errorQoverPRelative")),
      errDxDz_(iConfig.getParameter<double>("errorDxDz")),
      errDyDz_(iConfig.getParameter<double>("errorDyDz")),
      errX_(iConfig.getParameter<double>("errorX")),
      errY_(iConfig.getParameter<double>("errorY")),
      smear_(iConfig.getParameter<bool>("smear")),
      randomSeed_(iConfig.getParameter<unsigned int>("randomSeed")) {
  for (const auto& tag : iConfig.getParameter<std::vector<edm::InputTag>>("simHits"))
    simHitTokens_.emplace_back(consumes<edm::PSimHitContainer>(tag));

  produces<TrajectorySeedCollection>();
  produces<reco::TrackCollection>();
  produces<std::vector<int>>("simTrackIndex");
}

void SimHitJetCoreSeedGenerator::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto seeds = std::make_unique<TrajectorySeedCollection>();
  auto seedTracks = std::make_unique<reco::TrackCollection>();
  auto seedSimIdx = std::make_unique<std::vector<int>>();

  const auto& geom = iSetup.getData(geomToken_);
  const auto& tTopo = iSetup.getData(topoToken_);

  const auto& cores = iEvent.get(coresToken_);
  const auto& simTracks = iEvent.get(simTracksToken_);
  const auto& simVertices = iEvent.get(simVerticesToken_);

  // ---- select the jets first: nothing to do without a core ----
  std::vector<const reco::Candidate*> jets;
  for (const auto& jet : cores)
    if (jet.pt() > jetPtMin_ && std::abs(jet.eta()) < jetAbsEtaMax_)
      jets.push_back(&jet);

  if (!jets.empty()) {
    // ---- index SimTracks by (event, trackId) ----
    std::unordered_map<uint64_t, size_t> trackIndex;
    trackIndex.reserve(simTracks.size());
    for (size_t i = 0; i < simTracks.size(); ++i)
      trackIndex.emplace(trackKey(simTracks[i].eventId().rawId(), simTracks[i].trackId()), i);

    // ---- collect SimHits per SimTrack (only hits really made by that particle) ----
    std::unordered_map<size_t, std::vector<const PSimHit*>> hitsByTrack;
    std::vector<edm::Handle<edm::PSimHitContainer>> hitHandles(simHitTokens_.size());
    for (size_t t = 0; t < simHitTokens_.size(); ++t) {
      iEvent.getByToken(simHitTokens_[t], hitHandles[t]);
      for (const auto& sh : *hitHandles[t]) {
        auto it = trackIndex.find(trackKey(sh.eventId().rawId(), sh.trackId()));
        if (it == trackIndex.end())
          continue;
        // hits from secondaries that were not stored as SimTracks carry the parent's trackId
        if (sh.particleType() != simTracks[it->second].type())
          continue;
        hitsByTrack[it->second].push_back(&sh);
      }
    }

    // simulated primary vertex (first SimVertex of the signal event)
    GlobalPoint simPV(0, 0, 0);
    if (!simVertices.empty())
      simPV = GlobalPoint(simVertices[0].position().x(), simVertices[0].position().y(), simVertices[0].position().z());

    // reproducible per-event random engine (only used if smear_)
    std::mt19937_64 rng(uint64_t(randomSeed_) ^ (uint64_t(iEvent.id().run()) << 40) ^
                        (uint64_t(iEvent.id().luminosityBlock()) << 20) ^ uint64_t(iEvent.id().event()));
    std::normal_distribution<double> gaus(0., 1.);

    std::unordered_set<size_t> usedTracks;  // one seed per SimTrack

    for (const auto& [simIdx, hitsUnsorted] : hitsByTrack) {
      const SimTrack& st = simTracks[simIdx];

      // ---- SimTrack-level selection ----
      const int charge = static_cast<int>(std::lround(st.charge()));
      if (charge == 0)
        continue;
      if (st.momentum().pt() < simTrackPtMin_)
        continue;
      if (signalOnly_ && (st.eventId().bunchCrossing() != 0 || st.eventId().event() != 0))
        continue;
      if (maxProdRadius_ >= 0 && !st.noVertex()) {
        const int iv = st.vertIndex();
        if (iv >= 0 && iv < int(simVertices.size()) && simVertices[iv].position().Rho() > maxProdRadius_)
          continue;
      }

      // ---- hits in time order; drop hits after a catastrophic momentum loss ----
      std::vector<const PSimHit*> hits(hitsUnsorted);
      std::sort(hits.begin(), hits.end(), [](const PSimHit* a, const PSimHit* b) { return a->tof() < b->tof(); });
      const double pSim = st.momentum().P();
      hits.erase(std::remove_if(hits.begin(),
                                hits.end(),
                                [&](const PSimHit* h) { return h->pabs() < minHitMomentumFraction_ * pSim; }),
                 hits.end());
      if (hits.empty())
        continue;

      // ---- pixel-layer multiplicity ----
      std::unordered_set<unsigned int> pixLayers;
      for (const auto* h : hits) {
        const DetId id(h->detUnitId());
        if (isPixel(id))
          pixLayers.insert(id.subdetId() * 1000 + tTopo.side(id) * 100 + tTopo.layer(id));
      }
      if (pixLayers.size() < minPixelLayers_)
        continue;

      // ---- choose the seed hit ----
      const PSimHit* seedHit = nullptr;
      if (seedLayer_.subdet != 0) {
        for (const auto* h : hits) {
          const DetId id(h->detUnitId());
          if (int(id.subdetId()) == seedLayer_.subdet && tTopo.layer(id) == seedLayer_.layer) {
            seedHit = h;
            break;
          }
        }
      }
      if (seedHit == nullptr && (seedLayer_.subdet == 0 || fallbackToInnermost_))
        seedHit = hits.front();
      if (seedHit == nullptr)
        continue;

      const GeomDet* det = geom.idToDetUnit(DetId(seedHit->detUnitId()));
      if (det == nullptr)
        continue;

      const LocalPoint lpos(seedHit->localPosition().x(), seedHit->localPosition().y(), 0.f);
      const LocalVector lmom = seedHit->momentumAtEntry();
      const GlobalPoint gpos = det->surface().toGlobal(lpos);
      const GlobalVector gmom = det->surface().toGlobal(lmom);

      // ---- is this track in the core of any selected jet? ----
      const GlobalVector hitDir = gpos - simPV;
      bool inCore = false;
      for (const auto* jet : jets) {
        if (reco::deltaR(jet->eta(), jet->phi(), gmom.eta(), gmom.phi()) >= deltaR_)
          continue;
        if (hitDeltaR_ > 0 && reco::deltaR(jet->eta(), jet->phi(), hitDir.eta(), hitDir.phi()) >= hitDeltaR_)
          continue;
        inCore = true;
        break;
      }
      if (!inCore || !usedTracks.insert(simIdx).second)
        continue;

      // ---- seed parameters (q/p, dx/dz, dy/dz, x, y) at the module mid-plane ----
      LocalTrajectoryParameters truePar(lpos, lmom, TrackCharge(charge));
      const double qbp = truePar.qbp();
      const double sQbp = std::hypot(errQoverP_, errQoverPRel_ > 0 ? errQoverPRel_ * std::abs(qbp) : 0.);

      LocalTrajectoryParameters par = truePar;
      if (smear_) {
        const auto v = truePar.vector();  // (q/p, dxdz, dydz, x, y)
        double sq = v[0] + sQbp * gaus(rng);
        if (sq * v[0] <= 0)  // never flip the charge / cross zero
          sq = v[0];
        par = LocalTrajectoryParameters(sq,
                                        v[1] + errDxDz_ * gaus(rng),
                                        v[2] + errDyDz_ * gaus(rng),
                                        v[3] + errX_ * gaus(rng),
                                        v[4] + errY_ * gaus(rng),
                                        truePar.pzSign(),
                                        true);
      }

      // lower-triangular packed covariance, see LocalTrajectoryError
      float em[15] = {0};
      em[0] = sQbp * sQbp;          // q/p
      em[2] = errDxDz_ * errDxDz_;  // dx/dz
      em[5] = errDyDz_ * errDyDz_;  // dy/dz
      em[9] = errX_ * errX_;        // x
      em[14] = errY_ * errY_;       // y

      const GlobalVector seedMom = det->surface().toGlobal(par.momentum());
      const GlobalPoint seedPos = det->surface().toGlobal(par.position());

      seeds->emplace_back(PTrajectoryStateOnDet(par, seedMom.perp(), em, det->geographicalId().rawId(), 0),
                          TrajectorySeed::RecHitContainer(),
                          alongMomentum);

      reco::Track::CovarianceMatrix cov;
      seedTracks->emplace_back(0.,
                               0.,
                               reco::Track::Point(seedPos.x(), seedPos.y(), seedPos.z()),
                               reco::Track::Vector(seedMom.x(), seedMom.y(), seedMom.z()),
                               charge,
                               cov);
      seedSimIdx->push_back(int(simIdx));

      LogDebug("SimHitJetCoreSeedGenerator")
          << "seed from SimTrack " << st.trackId() << " pdg " << st.type() << " q " << charge << " pt "
          << seedMom.perp() << " eta " << seedMom.eta() << " phi " << seedMom.phi() << " on det "
          << det->geographicalId().rawId() << " (subdet " << det->geographicalId().subdetId() << ", layer "
          << tTopo.layer(det->geographicalId()) << ")";
    }
  }

  edm::LogInfo("SimHitJetCoreSeedGenerator") << jets.size() << " jet cores, " << seeds->size() << " truth seeds";

  iEvent.put(std::move(seeds));
  iEvent.put(std::move(seedTracks));
  iEvent.put(std::move(seedSimIdx), "simTrackIndex");
}

void SimHitJetCoreSeedGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("cores", edm::InputTag("jetsForCoreTracking"))
      ->setComment("jets (any reco::Candidate: CaloJets, PFJets, GenJets) defining the cores");
  desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("simVertices", edm::InputTag("g4SimHits"));
  desc.add<std::vector<edm::InputTag>>("simHits",
                                       {edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
                                        edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof")})
      ->setComment("PSimHit collections; add TIB/TID/TOB/TEC collections to allow strip seeds");

  desc.add<double>("jetPtMin", 100.);
  desc.add<double>("jetAbsEtaMax", 2.5);
  desc.add<double>("deltaR", 0.1)->setComment("max dR(jet, SimTrack momentum at the seed hit)");
  desc.add<double>("hitDeltaR", -1.)
      ->setComment("max dR(jet, seed hit position w.r.t. simulated PV); <=0 disables");
  desc.add<double>("simTrackPtMin", 0.5);
  desc.add<double>("maxProductionRadius", 10.)->setComment("cm, SimTrack production radius; <0 disables");
  desc.add<bool>("signalOnly", true);
  desc.add<unsigned int>("minPixelLayers", 1);

  desc.add<std::string>("seedLayer", "innermost")
      ->setComment("'innermost' (first hit in time), or a layer such as 'BPix2' (DeepCore-like), 'FPix1', 'TIB1'");
  desc.add<bool>("fallbackToInnermost", true)->setComment("if seedLayer is not crossed, use the innermost hit");
  desc.add<double>("minHitMomentumFraction", 0.)
      ->setComment("drop SimHits with |p| < frac * |p_SimTrack| (e.g. after hard brems/nuclear interaction)");

  // defaults = DeepCore NN residual widths (sqrt of the variances used in DeepCoreSeedGenerator)
  desc.add<double>("errorQoverP", 0.15);
  desc.add<double>("errorQoverPRelative", -1.);
  desc.add<double>("errorDxDz", std::sqrt(0.5e-5));
  desc.add<double>("errorDyDz", std::sqrt(0.5e-5));
  desc.add<double>("errorX", std::sqrt(2e-5));
  desc.add<double>("errorY", std::sqrt(2e-5));
  desc.add<bool>("smear", false)->setComment("Gaussian-smear seed parameters with the errors above");
  desc.add<unsigned int>("randomSeed", 12345);

  descriptions.add("simHitJetCoreSeedGenerator", desc);
}

DEFINE_FWK_MODULE(SimHitJetCoreSeedGenerator);
