// LostTrackAnalyzer
// -----------------
// One TTree entry per selected simulated track (TrackingParticle, default: signal, in-time, charged,
// pT > 100 GeV, |eta| < 2.5, production vertex rho < 2.5 cm and |z| < 30 cm).
// For each entry it records, for a list of reco collections, whether the TP was found:
//
//   hit-based (quickTrackAssociatorByHits, same definition as the MTV efficiency):
//       <name>_nMatch   number of associated objects (0 = not found, >1 = duplicates)
//       <name>_q        quality of the best association
//       <name>_pt, _nHit, _algo, _oalgo, _hp   of the best associated track
//   geometric (any object, needed for the hitless DeepCore seeds):
//       <name>_dR       min Delta R(object direction, TP direction)
//       <name>_dPtRel   (pT_obj - pT_TP)/pT_TP of that nearest object
//       <name>_dDz      dz_obj - dz_TP (w.r.t. beam spot) of that nearest object
//       <name>_nNear    number of objects with Delta R < nearDR
//
// Track collections ("tracks") are edm::View<reco::Track>: track iterations, jet-core chains,
// generalTracks and the seed tracks of TrackFromSeedProducer (seedTracks<seedLabel>).
// Candidate collections ("candidates") are TrackCandidateCollection (hit-based only).
//
// Per-TP information: kinematics, dxy/dz w.r.t. the beam spot, number of sim hits and layers,
// nearest jet (pT > jetPtMin), nearest other charged TP (pT > 1 GeV), and the number of its pixel
// clusters that are also associated to another TP (merged clusters that were not split).
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimTracker/TrackAssociation/interface/TrackingParticleIP.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "TTree.h"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
  std::string branchName(const edm::InputTag& t) {
    std::string s = t.label();
    if (!t.instance().empty())
      s += "_" + t.instance();
    return s;
  }
}  // namespace

class LostTrackAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit LostTrackAnalyzer(const edm::ParameterSet& cfg);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(const edm::Event& ev, const edm::EventSetup&) override;

private:
  struct Coll {  // per-collection output
    std::string name;
    int nMatch, algo, oalgo, nHit, hp, nNear;
    float q, pt, dR, dPtRel, dDz;
    void reset() {
      nMatch = 0;
      algo = oalgo = nHit = hp = -1;
      nNear = 0;
      q = pt = -1.f;
      dR = dPtRel = dDz = 999.f;
    }
  };

  // configuration
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> assocToken_;
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> jetToken_;
  edm::EDGetTokenT<ClusterTPAssociation> clusterTPToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int>> nLayToken_, nPixLayToken_;
  bool useJets_, useClusterTP_, useNLay_, useNPixLay_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> trkTokens_;
  std::vector<edm::EDGetTokenT<TrackCandidateCollection>> candTokens_;
  double ptMin_, absEtaMax_, tipMax_, lipMax_, jetPtMin_, nearDR_, otherTPPtMin_;
  bool signalOnly_, chargedOnly_;

  // output
  TTree* tree_;
  unsigned run_, lumi_, event_;
  int tpIdx_, pdgId_, charge_, nSimHits_, nLay_, nPixLay_, nPixClus_, nPixClusShared_, nPixClusSharedHighPt_;
  float pt_, eta_, phi_, dxy_, dz_, vrho_, vz_, jetDR_, jetPt_, jetEta_, tpDR_, tpDRpt_;
  std::vector<Coll> trk_, cand_;  // sized once in the constructor (branch addresses stay valid)
};

LostTrackAnalyzer::LostTrackAnalyzer(const edm::ParameterSet& cfg)
    : tpToken_(consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticles"))),
      assocToken_(consumes<reco::TrackToTrackingParticleAssociator>(cfg.getParameter<edm::InputTag>("associator"))),
      bsToken_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))),
      ptMin_(cfg.getParameter<double>("ptMin")),
      absEtaMax_(cfg.getParameter<double>("absEtaMax")),
      tipMax_(cfg.getParameter<double>("tipMax")),
      lipMax_(cfg.getParameter<double>("lipMax")),
      jetPtMin_(cfg.getParameter<double>("jetPtMin")),
      nearDR_(cfg.getParameter<double>("nearDR")),
      otherTPPtMin_(cfg.getParameter<double>("otherTPPtMin")),
      signalOnly_(cfg.getParameter<bool>("signalOnly")),
      chargedOnly_(cfg.getParameter<bool>("chargedOnly")) {
  usesResource("TFileService");
  auto const jets = cfg.getParameter<edm::InputTag>("jets");
  useJets_ = !jets.label().empty();
  if (useJets_)
    jetToken_ = consumes<edm::View<reco::Candidate>>(jets);
  auto const ctp = cfg.getParameter<edm::InputTag>("clusterTP");
  useClusterTP_ = !ctp.label().empty();
  if (useClusterTP_)
    clusterTPToken_ = consumes<ClusterTPAssociation>(ctp);
  auto const nl = cfg.getParameter<edm::InputTag>("tpNLayers");
  useNLay_ = !nl.label().empty();
  if (useNLay_)
    nLayToken_ = consumes<edm::ValueMap<unsigned int>>(nl);
  auto const npl = cfg.getParameter<edm::InputTag>("tpNPixelLayers");
  useNPixLay_ = !npl.label().empty();
  if (useNPixLay_)
    nPixLayToken_ = consumes<edm::ValueMap<unsigned int>>(npl);

  for (auto const& t : cfg.getParameter<std::vector<edm::InputTag>>("tracks")) {
    trkTokens_.push_back(consumes<edm::View<reco::Track>>(t));
    trk_.emplace_back();
    trk_.back().name = branchName(t);
  }
  for (auto const& t : cfg.getParameter<std::vector<edm::InputTag>>("candidates")) {
    candTokens_.push_back(consumes<TrackCandidateCollection>(t));
    cand_.emplace_back();
    cand_.back().name = branchName(t);
  }

  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("tp", "selected TrackingParticles and their reco matches");
  tree_->Branch("run", &run_);
  tree_->Branch("lumi", &lumi_);
  tree_->Branch("event", &event_);
  tree_->Branch("tpIdx", &tpIdx_);
  tree_->Branch("pdgId", &pdgId_);
  tree_->Branch("charge", &charge_);
  tree_->Branch("pt", &pt_);
  tree_->Branch("eta", &eta_);
  tree_->Branch("phi", &phi_);
  tree_->Branch("dxy", &dxy_);
  tree_->Branch("dz", &dz_);
  tree_->Branch("vrho", &vrho_);
  tree_->Branch("vz", &vz_);
  tree_->Branch("nSimHits", &nSimHits_);
  tree_->Branch("nLay", &nLay_);
  tree_->Branch("nPixLay", &nPixLay_);
  tree_->Branch("nPixClus", &nPixClus_);
  tree_->Branch("nPixClusShared", &nPixClusShared_);
  tree_->Branch("nPixClusSharedHighPt", &nPixClusSharedHighPt_);
  tree_->Branch("jetDR", &jetDR_);
  tree_->Branch("jetPt", &jetPt_);
  tree_->Branch("jetEta", &jetEta_);
  tree_->Branch("tpDR", &tpDR_);
  tree_->Branch("tpDRpt", &tpDRpt_);
  for (auto& c : trk_) {
    tree_->Branch((c.name + "_nMatch").c_str(), &c.nMatch);
    tree_->Branch((c.name + "_q").c_str(), &c.q);
    tree_->Branch((c.name + "_pt").c_str(), &c.pt);
    tree_->Branch((c.name + "_nHit").c_str(), &c.nHit);
    tree_->Branch((c.name + "_algo").c_str(), &c.algo);
    tree_->Branch((c.name + "_oalgo").c_str(), &c.oalgo);
    tree_->Branch((c.name + "_hp").c_str(), &c.hp);
    tree_->Branch((c.name + "_dR").c_str(), &c.dR);
    tree_->Branch((c.name + "_dPtRel").c_str(), &c.dPtRel);
    tree_->Branch((c.name + "_dDz").c_str(), &c.dDz);
    tree_->Branch((c.name + "_nNear").c_str(), &c.nNear);
  }
  for (auto& c : cand_) {
    tree_->Branch((c.name + "_nMatch").c_str(), &c.nMatch);
    tree_->Branch((c.name + "_q").c_str(), &c.q);
    tree_->Branch((c.name + "_nHit").c_str(), &c.nHit);
  }
}

void LostTrackAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("associator", edm::InputTag("quickTrackAssociatorByHits"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4CaloJetsForTrk"));
  desc.add<edm::InputTag>("clusterTP", edm::InputTag("tpClusterProducer"));
  desc.add<edm::InputTag>("tpNLayers", edm::InputTag("trackingParticleNumberOfLayersProducer", "trackerLayers"));
  desc.add<edm::InputTag>("tpNPixelLayers", edm::InputTag("trackingParticleNumberOfLayersProducer", "pixelLayers"));
  desc.add<std::vector<edm::InputTag>>("tracks", {edm::InputTag("generalTracks")});
  desc.add<std::vector<edm::InputTag>>("candidates", {});
  desc.add<double>("ptMin", 100.);
  desc.add<double>("absEtaMax", 2.5);
  desc.add<double>("tipMax", 2.5);
  desc.add<double>("lipMax", 30.);
  desc.add<double>("jetPtMin", 100.);
  desc.add<double>("nearDR", 0.02);
  desc.add<double>("otherTPPtMin", 1.);
  desc.add<bool>("signalOnly", true);
  desc.add<bool>("chargedOnly", true);
  descriptions.add("lostTrackAnalyzer", desc);
}

void LostTrackAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup&) {
  auto hTP = ev.getHandle(tpToken_);
  auto const& tps = *hTP;
  auto const& assoc = ev.get(assocToken_);
  auto const& bs = ev.get(bsToken_);

  // ---- select the TPs
  TrackingParticleRefVector sel;
  for (size_t i = 0; i < tps.size(); ++i) {
    auto const& tp = tps[i];
    if (signalOnly_ && (tp.eventId().bunchCrossing() != 0 || tp.eventId().event() != 0))
      continue;
    if (chargedOnly_ && tp.charge() == 0)
      continue;
    if (tp.pt() < ptMin_ || std::abs(tp.eta()) > absEtaMax_)
      continue;
    if (std::hypot(tp.vx(), tp.vy()) > tipMax_ || std::abs(tp.vz()) > lipMax_)
      continue;
    sel.push_back(TrackingParticleRef(hTP, i));
  }
  if (sel.empty())
    return;

  // ---- optional inputs
  edm::Handle<edm::View<reco::Candidate>> hJets;
  if (useJets_)
    hJets = ev.getHandle(jetToken_);
  edm::Handle<edm::ValueMap<unsigned int>> hNLay, hNPixLay;
  if (useNLay_)
    hNLay = ev.getHandle(nLayToken_);
  if (useNPixLay_)
    hNPixLay = ev.getHandle(nPixLayToken_);

  // pixel clusters: cluster -> number of TPs (all, and with pT > otherTPPtMin); TP -> its clusters
  std::unordered_map<unsigned, unsigned> nTPofClus, nHighPtTPofClus;
  std::unordered_map<unsigned, std::vector<unsigned>> clusOfTP;
  bool haveClusTP = false;
  if (useClusterTP_) {
    auto hCTP = ev.getHandle(clusterTPToken_);
    if (hCTP.isValid()) {
      haveClusTP = true;
      std::unordered_map<unsigned, bool> isSel;
      for (auto const& r : sel)
        isSel[r.key()] = true;
      for (auto const& p : *hCTP) {
        if (!p.first.isPixel() || !p.first.isValid() || p.second.id() != hTP.id())
          continue;
        unsigned const c = p.first.rawIndex();
        ++nTPofClus[c];
        if (p.second->pt() > otherTPPtMin_)
          ++nHighPtTPofClus[c];
        if (isSel.count(p.second.key()))
          clusOfTP[p.second.key()].push_back(c);
      }
    }
  }

  // ---- associations
  std::vector<edm::Handle<edm::View<reco::Track>>> hTrk(trkTokens_.size());
  std::vector<reco::SimToRecoCollection> s2rTrk(trkTokens_.size());
  for (size_t c = 0; c < trkTokens_.size(); ++c) {
    hTrk[c] = ev.getHandle(trkTokens_[c]);
    if (hTrk[c].isValid())
      s2rTrk[c] = assoc.associateSimToReco(hTrk[c], hTP);
  }
  std::vector<edm::Handle<TrackCandidateCollection>> hCand(candTokens_.size());
  std::vector<reco::SimToRecoCollectionTCandidate> s2rCand(candTokens_.size());
  for (size_t c = 0; c < candTokens_.size(); ++c) {
    hCand[c] = ev.getHandle(candTokens_[c]);
    if (hCand[c].isValid())
      s2rCand[c] = assoc.associateSimToReco(hCand[c], sel);
  }

  run_ = ev.id().run();
  lumi_ = ev.id().luminosityBlock();
  event_ = ev.id().event();
  double const nearDR2 = nearDR_ * nearDR_;

  for (auto const& tpr : sel) {
    auto const& tp = *tpr;
    tpIdx_ = tpr.key();
    pdgId_ = tp.pdgId();
    charge_ = tp.threeCharge() / 3;
    pt_ = tp.pt();
    eta_ = tp.eta();
    phi_ = tp.phi();
    dxy_ = TrackingParticleIP::dxy(tp.vertex(), tp.momentum(), bs.position());
    dz_ = TrackingParticleIP::dz(tp.vertex(), tp.momentum(), bs.position());
    vrho_ = std::hypot(tp.vx(), tp.vy());
    vz_ = tp.vz();
    nSimHits_ = tp.numberOfTrackerHits();
    nLay_ = (hNLay.isValid() ? (int)(*hNLay)[tpr] : -1);
    nPixLay_ = (hNPixLay.isValid() ? (int)(*hNPixLay)[tpr] : -1);

    nPixClus_ = nPixClusShared_ = nPixClusSharedHighPt_ = -1;
    if (haveClusTP) {
      nPixClus_ = nPixClusShared_ = nPixClusSharedHighPt_ = 0;
      auto it = clusOfTP.find(tpr.key());
      if (it != clusOfTP.end())
        for (unsigned c : it->second) {
          ++nPixClus_;
          if (nTPofClus[c] > 1)
            ++nPixClusShared_;
          if (nHighPtTPofClus[c] > 1)  // this TP (pT>ptMin>otherTPPtMin) + another TP above otherTPPtMin
            ++nPixClusSharedHighPt_;
        }
    }

    // nearest jet
    jetDR_ = 999.f;
    jetPt_ = jetEta_ = -1.f;
    if (hJets.isValid())
      for (auto const& j : *hJets) {
        if (j.pt() < jetPtMin_)
          continue;
        float const d = reco::deltaR(double(j.eta()), double(j.phi()), double(eta_), double(phi_));
        if (d < jetDR_) {
          jetDR_ = d;
          jetPt_ = j.pt();
          jetEta_ = j.eta();
        }
      }

    // nearest other charged signal TP
    tpDR_ = 999.f;
    tpDRpt_ = -1.f;
    for (size_t k = 0; k < tps.size(); ++k) {
      if (k == tpr.key())
        continue;
      auto const& o = tps[k];
      if (o.charge() == 0 || o.pt() < otherTPPtMin_ || o.eventId().bunchCrossing() != 0 || o.eventId().event() != 0)
        continue;
      float const d = reco::deltaR(double(o.eta()), double(o.phi()), double(eta_), double(phi_));
      if (d < tpDR_) {
        tpDR_ = d;
        tpDRpt_ = o.pt();
      }
    }

    // track collections
    for (size_t c = 0; c < trk_.size(); ++c) {
      auto& out = trk_[c];
      out.reset();
      if (!hTrk[c].isValid()) {
        out.nMatch = -1;  // collection missing in this event
        continue;
      }
      auto f = s2rTrk[c].find(tpr);
      if (f != s2rTrk[c].end() && !f->val.empty()) {
        out.nMatch = f->val.size();
        auto const& best = f->val.front();  // sorted by quality
        out.q = best.second;
        out.pt = best.first->pt();
        out.nHit = best.first->numberOfValidHits();
        out.algo = best.first->algo();
        out.oalgo = best.first->originalAlgo();
        out.hp = best.first->quality(reco::TrackBase::highPurity);
      }
      double best2 = 1e9;
      for (auto const& t : *hTrk[c]) {
        double const d2 = reco::deltaR2(double(t.eta()), double(t.phi()), double(eta_), double(phi_));
        if (d2 < nearDR2)
          ++out.nNear;
        if (d2 < best2) {
          best2 = d2;
          out.dPtRel = (t.pt() - pt_) / pt_;
          out.dDz = t.dz(bs.position()) - dz_;
        }
      }
      if (best2 < 1e9)
        out.dR = std::sqrt(best2);
    }

    // candidate collections
    for (size_t c = 0; c < cand_.size(); ++c) {
      auto& out = cand_[c];
      out.reset();
      if (!hCand[c].isValid()) {
        out.nMatch = -1;
        continue;
      }
      auto f = s2rCand[c].find(tpr);
      if (f != s2rCand[c].end() && !f->val.empty()) {
        out.nMatch = f->val.size();
        out.q = f->val.front().second;
        out.nHit = f->val.front().first->nRecHits();
      }
    }

    tree_->Fill();
  }
}

DEFINE_FWK_MODULE(LostTrackAnalyzer);
