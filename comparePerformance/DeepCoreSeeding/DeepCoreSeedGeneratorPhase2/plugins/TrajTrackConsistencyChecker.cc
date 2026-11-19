// Diagnostic: for each track collection that also stores trajectories, check what
// TrackCollectionCloner (DuplicateListMerger / TrackCollectionMerger) assumes:
//   (1) tracks and trajectories are one-to-one, in the same order
//   (2) trajectory.measurements().size() == track.recHitsSize()
// and print the offending tracks (algo, pT, eta, #measurements, #hits, measurements without a det).
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include <sstream>
#include <vector>

class TrajTrackConsistencyChecker : public edm::global::EDAnalyzer<> {
public:
  explicit TrajTrackConsistencyChecker(const edm::ParameterSet& cfg) : maxPrint_(cfg.getParameter<unsigned>("maxPrint")) {
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("collections")) {
      labels_.push_back(tag.encode());
      trk_.push_back(consumes<reco::TrackCollection>(tag));
      traj_.push_back(consumes<std::vector<Trajectory>>(tag));
      ass_.push_back(consumes<TrajTrackAssociationCollection>(tag));
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<edm::InputTag>>("collections", {});
    desc.add<unsigned>("maxPrint", 10);
    descriptions.add("trajTrackConsistencyChecker", desc);
  }

  void analyze(edm::StreamID, const edm::Event& ev, const edm::EventSetup&) const override {
    for (unsigned c = 0; c < labels_.size(); ++c) {
      auto hTrk = ev.getHandle(trk_[c]);
      auto hTraj = ev.getHandle(traj_[c]);
      auto hAss = ev.getHandle(ass_[c]);
      std::ostringstream out;
      out << "[TTCheck] event " << ev.id().event() << "  " << labels_[c] << ": ";
      if (!hTrk.isValid()) {
        out << "no tracks in event";
        edm::LogPrint("TTCheck") << out.str();
        continue;
      }
      out << hTrk->size() << " tracks, ";
      if (!hTraj.isValid()) {
        out << "NO trajectories";
        edm::LogPrint("TTCheck") << out.str();
        continue;
      }
      out << hTraj->size() << " trajectories, " << (hAss.isValid() ? hAss->size() : 0) << " associations";
      if (hTraj->size() != hTrk->size())
        out << "  <-- NOT one-to-one";

      // (a) index assumption of TrackCollectionCloner: trajectory k <-> track k
      unsigned badIdx = 0, printed = 0;
      unsigned n = std::min(hTraj->size(), hTrk->size());
      for (unsigned k = 0; k < n; ++k) {
        auto const& t = (*hTraj)[k];
        auto const& tk = (*hTrk)[k];
        if (t.measurements().size() == tk.recHitsSize())
          continue;
        ++badIdx;
        if (printed++ < maxPrint_) {
          unsigned noDet = 0, invalid = 0;
          for (auto const& m : t.measurements()) {
            if (m.recHitR().det() == nullptr)
              ++noDet;
            if (!m.recHitR().isValid())
              ++invalid;
          }
          out << "\n     idx " << k << " algo " << reco::TrackBase::algoName(tk.algo()) << " orig "
              << reco::TrackBase::algoName(tk.originalAlgo()) << " pt " << tk.pt() << " eta " << tk.eta()
              << "  meas " << t.measurements().size() << " (noDet " << noDet << ", invalid " << invalid << ")"
              << "  hits " << tk.recHitsSize() << "  seedHits " << t.seed().nHits();
        }
      }
      // (b) the same through the association map (tells index shift from genuine mismatch)
      unsigned badAss = 0, shifted = 0;
      if (hAss.isValid())
        for (auto const& p : *hAss) {
          if (p.key.key() != p.val.key())
            ++shifted;
          if (p.key->measurements().size() != p.val->recHitsSize())
            ++badAss;
        }
      out << "\n     mismatches by index: " << badIdx << "   by association: " << badAss
          << "   associations with traj index != track index: " << shifted;
      edm::LogPrint("TTCheck") << out.str();
    }
  }

private:
  std::vector<std::string> labels_;
  std::vector<edm::EDGetTokenT<reco::TrackCollection>> trk_;
  std::vector<edm::EDGetTokenT<std::vector<Trajectory>>> traj_;
  std::vector<edm::EDGetTokenT<TrajTrackAssociationCollection>> ass_;
  unsigned maxPrint_;
};

DEFINE_FWK_MODULE(TrajTrackConsistencyChecker);
