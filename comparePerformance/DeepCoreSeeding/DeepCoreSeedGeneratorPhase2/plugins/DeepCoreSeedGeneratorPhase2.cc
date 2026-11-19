// -*- C++ -*-
//
// Package:    DeepCoreSeeding/DeepCoreSeedGeneratorPhase2
// Class:      DeepCoreSeedGeneratorPhase2
//
// DeepCore seed generator for the Phase-2 Inner Tracker, with ONNX Runtime inference of the PyTorch model
// (training/pytorch, exported with DeepCore_training_pytorch.py --exportONNX).
//
// The input image and the meaning of the network outputs follow DeepCoreNtuplizerPhase2 exactly:
//   - window of 120 x 45 pixels of 25 x 100 um around the crossing of the direction with each barrel layer
//     (same physical size as the Phase-1 30 x 30 window); modules with another pitch are skipped
//   - directions: merged clusters on TBPX layer 1 (layer 2 if none), plus the jet direction
//   - merged-cluster test as in JetCoreClusterSplitter, Phase-2 parameters
//   - forward-only line/plane intersection; module per layer = smallest distance of the crossing from the
//     module centre
//   - image charge = adc / chargeNorm, x index mirrored on "flipped" modules
//   - outputs: par[x][y][o] = (100*dx[cm], 100*dy[cm], 100*deta, 100*dphi, pt[GeV], flag), dx/dy relative to the
//     centre of the pixel (localX(i+0.5)) in the mirrored frame, deta/dphi relative to the window direction;
//     prob[x][y][o][0] after the sigmoid
//
// Original Phase-1 author: Valerio Bertacchi. Phase-2/ONNX version: DeepCore 2.4.
//
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

using cms::Ort::ONNXRuntime;

class DeepCoreSeedGeneratorPhase2 : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit DeepCoreSeedGeneratorPhase2(const edm::ParameterSet&, const ONNXRuntime*);
  ~DeepCoreSeedGeneratorPhase2() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*) {}

  // window geometry: must match DeepCoreNtuplizerPhase2 and the trained model
  static constexpr int jetDimX = 120;
  static constexpr int jetDimY = 45;
  static constexpr int Nlayer = 4;  // TBPX layers 1-4
  static constexpr int Nover = 3;
  static constexpr int Npar = 5;
  static constexpr int NparOut = Npar + 1;  // the model also predicts the "zero flag" column
  static constexpr int NprobOut = 2;        // only [..., 0] is used
  static constexpr int imageSize = jetDimX * jetDimY * Nlayer;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  struct Window {
    GlobalVector dir;
    const GeomDet* globDet = nullptr;  // module on layer 2, where the seeds are defined
    LocalPoint jetInter;               // crossing of dir with globDet (local frame)
    bool l2off = false;                // no charge on layer 2 in the window
    bool l134off = false;              // no charge on layers 1, 3 and 4
    int index = 0;                     // direction index within the jet
  };

  std::pair<bool, Basic3DVector<float>> findIntersection(const GlobalVector&,
                                                         const reco::Candidate::Point&,
                                                         const GeomDet*) const;
  bool pitchOK(const GeomDet*) const;
  std::pair<int, int> local2Pixel(double, double, const GeomDet*) const;
  int pixelFlipper(const GeomDet*) const;
  void fillPixelMatrix(const SiPixelCluster&, int layer, const LocalPoint& inter, const GeomDet*, float* image) const;
  const GeomDet* detectorSelector(int layer,
                                  const GlobalVector& dir,
                                  const reco::Vertex&,
                                  const TrackerTopology*,
                                  const edmNew::DetSetVector<SiPixelCluster>&) const;
  std::vector<GlobalVector> splittedClusterDirections(const reco::Candidate&,
                                                      const TrackerTopology*,
                                                      const PixelClusterParameterEstimator*,
                                                      const reco::Vertex&,
                                                      int layer,
                                                      const edmNew::DetSetVector<SiPixelCluster>&) const;
  void dumpWindow(const edm::EventID&, int jetIndex, const reco::Candidate&, const Window&, const float* image,
                  const float* prob, const float* par) const;

  // ---------- tokens ----------
  const edm::EDGetTokenT<reco::VertexCollection> vertices_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusters_;
  const edm::EDGetTokenT<edm::View<reco::Candidate>> cores_;
  // clusters used by tracking (after splitting), for the hit attached to the seed (attachHit)
  const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> hitClusters_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geometryToken_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> pixelCPEToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const GlobalTrackingGeometry* geometry_ = nullptr;

  // ---------- configuration ----------
  const double ptMin_;
  const double absEtaMax_;
  const double deltaR_;
  const double chargeFracMin_;
  const double centralMIPCharge_;
  const double expSizeXAtLorentzAngleIncidence_;
  const double expSizeXDeltaPerTanAlpha_;
  const double expSizeYAtNormalIncidence_;
  const double tanLorentzAngle_;
  const double tanLorentzAngleBarrelLayer1_;
  const float chargeNorm_;
  const double pitchX_;
  const double pitchY_;
  const std::vector<std::string> inputNames_;
  const std::vector<std::string> outputNames_;
  const double probThr_;
  const std::vector<double> dth_;   // per-overlap threshold offsets
  const std::vector<double> dthl_;  // extra offsets when layers 1, 3 and 4 are empty
  const double l2offDelta_;         // extra offset when layer 2 is empty
  const double seedPtMin_;
  const std::vector<double> seedErrors_;  // variances of (q/pt, dxdz, dydz, x, y)
  const unsigned int maxBatch_;
  // duplicate removal over the whole event (greedy, highest probability first); off if dupMaxDist <= 0
  const double dupMaxDist_;  // 3D distance of the seed points on the layer-2 module [cm]
  const double dupMaxDEta_;
  const double dupMaxDPhi_;
  // seed charge: +1 (default, as in Phase-1), -1, or 0 = one seed with each charge per prediction
  const int seedCharge_;
  // if > 0: q/p uncertainty of each seed = seedQoverPRelError * |q/p| (instead of seedErrors[0])
  const double seedQoverPRelError_;
  // attach the nearest (split) pixel cluster on the layer-2 module to the seed, and start from its position
  const bool attachHit_;
  const double attachMaxDX_;  // [cm], local x
  const double attachMaxDY_;  // [cm], local y
  const bool attachHitRequired_;  // drop the prediction if no cluster is found (else keep it without hit)
  const std::string dumpDir_;
};

// =================================================================================================

DeepCoreSeedGeneratorPhase2::DeepCoreSeedGeneratorPhase2(const edm::ParameterSet& cfg, const ONNXRuntime*)
    : vertices_(consumes(cfg.getParameter<edm::InputTag>("vertices"))),
      pixelClusters_(consumes(cfg.getParameter<edm::InputTag>("pixelClusters"))),
      cores_(consumes(cfg.getParameter<edm::InputTag>("cores"))),
      hitClusters_(consumes(cfg.getParameter<edm::InputTag>("attachHitClusters"))),
      geometryToken_(esConsumes()),
      pixelCPEToken_(esConsumes(edm::ESInputTag("", cfg.getParameter<std::string>("pixelCPE")))),
      topoToken_(esConsumes()),
      ptMin_(cfg.getParameter<double>("ptMin")),
      absEtaMax_(cfg.getParameter<double>("absEtaMax")),
      deltaR_(cfg.getParameter<double>("deltaR")),
      chargeFracMin_(cfg.getParameter<double>("chargeFractionMin")),
      centralMIPCharge_(cfg.getParameter<double>("centralMIPCharge")),
      expSizeXAtLorentzAngleIncidence_(cfg.getParameter<double>("expSizeXAtLorentzAngleIncidence")),
      expSizeXDeltaPerTanAlpha_(cfg.getParameter<double>("expSizeXDeltaPerTanAlpha")),
      expSizeYAtNormalIncidence_(cfg.getParameter<double>("expSizeYAtNormalIncidence")),
      tanLorentzAngle_(cfg.getParameter<double>("tanLorentzAngle")),
      tanLorentzAngleBarrelLayer1_(cfg.getParameter<double>("tanLorentzAngleBarrelLayer1")),
      chargeNorm_(cfg.getParameter<double>("chargeNorm")),
      pitchX_(cfg.getParameter<double>("pitchX")),
      pitchY_(cfg.getParameter<double>("pitchY")),
      inputNames_(cfg.getParameter<std::vector<std::string>>("inputTensorName")),
      outputNames_(cfg.getParameter<std::vector<std::string>>("outputTensorName")),
      probThr_(cfg.getParameter<double>("probThr")),
      dth_(cfg.getParameter<std::vector<double>>("probThrDeltaOverlap")),
      dthl_(cfg.getParameter<std::vector<double>>("probThrDeltaL134off")),
      l2offDelta_(cfg.getParameter<double>("probThrDeltaL2off")),
      seedPtMin_(cfg.getParameter<double>("seedPtMin")),
      seedErrors_(cfg.getParameter<std::vector<double>>("seedErrors")),
      maxBatch_(cfg.getParameter<unsigned int>("maxBatch")),
      dupMaxDist_(cfg.getParameter<double>("duplicateMaxDist")),
      dupMaxDEta_(cfg.getParameter<double>("duplicateMaxDEta")),
      dupMaxDPhi_(cfg.getParameter<double>("duplicateMaxDPhi")),
      seedCharge_(cfg.getParameter<int>("seedCharge")),
      seedQoverPRelError_(cfg.getParameter<double>("seedQoverPRelError")),
      attachHit_(cfg.getParameter<bool>("attachHit")),
      attachMaxDX_(cfg.getParameter<double>("attachMaxDX")),
      attachMaxDY_(cfg.getParameter<double>("attachMaxDY")),
      attachHitRequired_(cfg.getParameter<bool>("attachHitRequired")),
      dumpDir_(cfg.getUntrackedParameter<std::string>("dumpDir")) {
  if (seedCharge_ < -1 || seedCharge_ > 1)
    throw cms::Exception("Configuration") << "seedCharge must be -1, 0 (both) or +1";
  if (inputNames_.size() != 3 || outputNames_.size() != 2)
    throw cms::Exception("Configuration") << "inputTensorName needs 3 names and outputTensorName 2";
  if ((int)dth_.size() != Nover || (int)dthl_.size() != Nover || seedErrors_.size() != 5)
    throw cms::Exception("Configuration") << "probThrDeltaOverlap and probThrDeltaL134off need " << Nover
                                          << " values, seedErrors 5";
  produces<TrajectorySeedCollection>();
  produces<reco::TrackCollection>();
}

std::unique_ptr<ONNXRuntime> DeepCoreSeedGeneratorPhase2::initializeGlobalCache(const edm::ParameterSet& cfg) {
  return std::make_unique<ONNXRuntime>(cfg.getParameter<edm::FileInPath>("weightFile").fullPath());
}

// =================================================================================================

void DeepCoreSeedGeneratorPhase2::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto result = std::make_unique<TrajectorySeedCollection>();
  auto resultTracks = std::make_unique<reco::TrackCollection>();

  geometry_ = &iSetup.getData(geometryToken_);
  const auto& clusters = iEvent.get(pixelClusters_);
  const auto& vertices = iEvent.get(vertices_);
  const auto& cores = iEvent.get(cores_);
  const PixelClusterParameterEstimator* pixelCPE = &iSetup.getData(pixelCPEToken_);
  const TrackerTopology* tTopo = &iSetup.getData(topoToken_);
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> hHitClusters;
  if (attachHit_)
    hHitClusters = iEvent.getHandle(hitClusters_);

  if (vertices.empty()) {
    iEvent.put(std::move(result));
    iEvent.put(std::move(resultTracks));
    return;
  }
  const reco::Vertex& jetVertex = vertices[0];

  // seeds are collected first and written after the event-wide duplicate removal
  struct SeedCand {
    float prob;
    GlobalPoint point;
    float eta, phi;
    PTrajectoryStateOnDet state;
    GlobalVector dir;
    int charge;
    std::optional<SiPixelRecHit> hit;
  };
  std::vector<SeedCand> cands;
  const reco::Candidate::Point vtxPoint(jetVertex.position());

  for (unsigned int ji = 0; ji < cores.size(); ++ji) {
    const reco::Candidate& jet = cores[ji];
    if (jet.pt() < ptMin_ || std::abs(jet.eta()) > absEtaMax_)
      continue;

    // ---- window directions: merged clusters on layer 1 (layer 2 if none) + the jet direction ----
    std::vector<GlobalVector> dirs = splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 1, clusters);
    if (dirs.empty())
      dirs = splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 2, clusters);
    dirs.emplace_back(jet.px(), jet.py(), jet.pz());

    // ---- build the images ----
    std::vector<Window> windows;
    std::vector<float> images;  // [nWindows][X][Y][L], channels last as in the ntuple and the ONNX input
    windows.reserve(dirs.size());
    images.reserve(dirs.size() * imageSize);
    for (int cc = 0; cc < (int)dirs.size(); ++cc) {
      Window w;
      w.dir = dirs[cc];
      w.index = cc;
      w.globDet = detectorSelector(2, w.dir, jetVertex, tTopo, clusters);
      if (w.globDet == nullptr)
        continue;
      auto inter2 = findIntersection(w.dir, vtxPoint, w.globDet);
      if (!inter2.first)
        continue;
      w.jetInter = w.globDet->specificSurface().toLocal(GlobalPoint(inter2.second));

      std::vector<const GeomDet*> goodDets;
      for (int l = 1; l <= Nlayer; ++l) {
        const GeomDet* d = (l == 2) ? w.globDet : detectorSelector(l, w.dir, jetVertex, tTopo, clusters);
        if (d != nullptr && std::find(goodDets.begin(), goodDets.end(), d) == goodDets.end())
          goodDets.push_back(d);
      }

      const size_t offset = images.size();
      images.resize(offset + imageSize, 0.f);
      float* image = images.data() + offset;
      for (const GeomDet* det : goodDets) {
        if (!pitchOK(det))
          continue;
        auto dsIt = clusters.find(det->geographicalId().rawId());
        if (dsIt == clusters.end())
          continue;
        auto interPair = findIntersection(w.dir, vtxPoint, det);
        if (!interPair.first)
          continue;
        const LocalPoint localInter = det->specificSurface().toLocal(GlobalPoint(interPair.second));
        const int lay = tTopo->layer(det->geographicalId());
        const GeomDetUnit* detUnit = geometry_->idToDetUnit(det->geographicalId());
        for (const auto& aCluster : *dsIt) {
          const LocalPoint cPos = pixelCPE->localParametersV(aCluster, *detUnit)[0].first;
          if (std::abs(cPos.x() - localInter.x()) / pitchX_ > jetDimX / 2)
            continue;
          if (std::abs(cPos.y() - localInter.y()) / pitchY_ > jetDimY / 2)
            continue;
          fillPixelMatrix(aCluster, lay, localInter, det, image);
        }
      }
      // empty-layer flags, as in DeepCore_testing_pytorch.py (from the image itself)
      float q[Nlayer] = {0, 0, 0, 0};
      for (int i = 0; i < jetDimX * jetDimY; ++i)
        for (int l = 0; l < Nlayer; ++l)
          q[l] += image[i * Nlayer + l];
      w.l2off = !(q[1] > 0.f);
      w.l134off = !(q[0] > 0.f) && !(q[2] > 0.f) && !(q[3] > 0.f);
      windows.push_back(w);
    }
    if (windows.empty())
      continue;

    // ---- inference, in batches of up to maxBatch windows ----
    std::set<std::tuple<long, long, long, long, unsigned int>> ids;
    for (size_t b0 = 0; b0 < windows.size(); b0 += std::max(1u, maxBatch_)) {
      const size_t nn = std::min<size_t>(std::max(1u, maxBatch_), windows.size() - b0);
      const int64_t n = nn;
      cms::Ort::FloatArrays input(3);
      input[0].assign(nn, (float)jet.eta());
      input[1].assign(nn, (float)jet.pt());
      input[2].assign(images.begin() + b0 * imageSize, images.begin() + (b0 + nn) * imageSize);
      const std::vector<std::vector<int64_t>> shapes = {{n, 1}, {n, 1}, {n, jetDimX, jetDimY, Nlayer}};
      cms::Ort::FloatArrays out = globalCache()->run(inputNames_, input, shapes, outputNames_, n);
      const size_t perPar = (size_t)jetDimX * jetDimY * Nover * NparOut;
      const size_t perProb = (size_t)jetDimX * jetDimY * Nover * NprobOut;
      if (out.size() != 2 || out[0].size() != nn * perPar || out[1].size() != nn * perProb)
        throw cms::Exception("DeepCoreSeedGeneratorPhase2")
            << "unexpected ONNX output sizes " << (out.size() > 0 ? out[0].size() : 0) << ", "
            << (out.size() > 1 ? out[1].size() : 0) << " (expected " << nn * perPar << ", " << nn * perProb
            << "): is the model trained on a " << jetDimX << "x" << jetDimY << " window?";

      for (size_t k = 0; k < nn; ++k) {
        const Window& w = windows[b0 + k];
        const float* par = out[0].data() + k * perPar;
        const float* prob = out[1].data() + k * perProb;
        if (!dumpDir_.empty())
          dumpWindow(iEvent.id(), ji, jet, w, images.data() + (b0 + k) * imageSize, prob, par);

        const GeomDet* globDet = w.globDet;
        const auto* globPixDet = dynamic_cast<const PixelGeomDetUnit*>(globDet);
        if (globPixDet == nullptr)
          continue;
        const auto& topo = globPixDet->specificTopology();
        const int flip = pixelFlipper(globDet);
        // pixel of the window centre in the mirrored frame, as in DeepCoreNtuplizerPhase2::fillTrackInfo
        const std::pair<int, int> pixJ = local2Pixel(flip * w.jetInter.x(), w.jetInter.y(), globDet);

        for (int x = 0; x < jetDimX; ++x) {
          for (int y = 0; y < jetDimY; ++y) {
            for (int o = 0; o < Nover; ++o) {
              const size_t cell = ((size_t)x * jetDimY + y) * Nover + o;
              const float p = prob[cell * NprobOut];
              const double thr =
                  probThr_ + dth_[o] + (w.l2off ? l2offDelta_ : 0.) + (w.l134off ? dthl_[o] : 0.);
              if (p <= thr)
                continue;
              const float* tp = par + cell * NparOut;

              // position on the layer-2 module: centre of the pixel + predicted offset, back to the true frame
              const int ipx = x + pixJ.first - jetDimX / 2;
              const int ipy = y + pixJ.second - jetDimY / 2;
              const double u = topo.localX(ipx + 0.5f) + tp[0] * 0.01;  // mirrored frame
              const double lx = flip * u;
              const double ly = topo.localY(ipy + 0.5f) + tp[1] * 0.01;
              const LocalPoint localSeedPoint(lx, ly, 0.);

              const double trackEta = w.dir.eta() + tp[2] * 0.01;
              const double trackPhi = w.dir.phi() + tp[3] * 0.01;
              const double pt = std::max<double>(tp[4], seedPtMin_);
              const double theta = 2. * std::atan(std::exp(-trackEta));
              const GlobalVector globSeedDir(
                  GlobalVector::Polar(Geom::Theta<double>(theta), Geom::Phi<double>(trackPhi), pt / std::sin(theta)));
              const LocalVector localSeedDir = globDet->surface().toLocal(globSeedDir);

              // remove duplicates (same position within ~half a pixel and same direction within 2.5 mrad)
              const auto key = std::make_tuple(std::lround(lx / (0.5 * pitchX_)),
                                               std::lround(ly / (0.5 * pitchY_)),
                                               std::lround(trackEta * 400.),
                                               std::lround(trackPhi * 400.),
                                               globDet->geographicalId().rawId());
              if (!ids.insert(key).second)
                continue;

              // optional: attach the nearest tracking cluster on this module and start from its position
              std::optional<SiPixelRecHit> hit;
              LocalPoint startPoint = localSeedPoint;
              if (attachHit_ && hHitClusters.isValid()) {
                auto hs = hHitClusters->find(globDet->geographicalId().rawId());
                if (hs != hHitClusters->end()) {
                  double best = 1e9;
                  const SiPixelCluster* bestCl = nullptr;
                  for (const auto& cl : *hs) {
                    const LocalPoint cp = pixelCPE->localParametersV(cl, *globPixDet)[0].first;
                    const double dx = std::abs(cp.x() - lx), dy = std::abs(cp.y() - ly);
                    if (dx > attachMaxDX_ || dy > attachMaxDY_)
                      continue;
                    const double d = (dx / attachMaxDX_) * (dx / attachMaxDX_) + (dy / attachMaxDY_) * (dy / attachMaxDY_);
                    if (d < best) {
                      best = d;
                      bestCl = &cl;
                    }
                  }
                  if (bestCl != nullptr) {
                    auto hp = pixelCPE->getParameters(*bestCl, *globPixDet);
                    hit.emplace(std::get<0>(hp), std::get<1>(hp), std::get<2>(hp), *globDet,
                                edmNew::makeRefTo(hHitClusters, bestCl));
                    startPoint = LocalPoint(std::get<0>(hp).x(), std::get<0>(hp).y(), 0.);
                  }
                }
                if (!hit && attachHitRequired_)
                  continue;
              }

              float em[15] = {0};  // LocalTrajectoryError lower triangle: (q/p, dxdz, dydz, x, y)
              em[0] = seedQoverPRelError_ > 0. ? std::pow(seedQoverPRelError_ * std::sin(theta) / pt, 2)
                                               : seedErrors_[0];
              em[2] = seedErrors_[1];
              em[5] = seedErrors_[2];
              em[9] = seedErrors_[3];
              em[14] = seedErrors_[4];
              for (int q : {+1, -1}) {
                if (seedCharge_ != 0 && q != seedCharge_)
                  continue;
                const LocalTrajectoryParameters localParam(startPoint, localSeedDir, TrackCharge(q));
                cands.push_back({p,
                                 globDet->surface().toGlobal(startPoint),
                                 (float)trackEta,
                                 (float)trackPhi,
                                 PTrajectoryStateOnDet(localParam, pt, em, globDet->geographicalId().rawId(), 0),
                                 globSeedDir,
                                 q,
                                 hit});
              }
            }
          }
        }
      }
    }
  }
  // ---- event-wide duplicate removal (greedy non-maximum suppression in (eta, phi, position)) ----
  std::vector<unsigned> order(cands.size());
  for (unsigned i = 0; i < order.size(); ++i)
    order[i] = i;
  std::vector<unsigned> kept;
  if (dupMaxDist_ > 0. && !cands.empty()) {
    std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return cands[a].prob > cands[b].prob; });
    // grid in (eta, phi) with cells of size (dEta, dPhi): only the 3x3 neighbouring cells need to be checked
    const double ce = std::max(dupMaxDEta_, 1e-4), cp = std::max(dupMaxDPhi_, 1e-4);
    const long nPhiCells = std::max(1L, (long)std::floor(2. * M_PI / cp));
    auto cellOf = [&](const SeedCand& c) {
      long ie = std::lround(std::floor(c.eta / ce));
      long ip = (long)std::floor((c.phi + M_PI) / (2. * M_PI) * nPhiCells);
      ip = ((ip % nPhiCells) + nPhiCells) % nPhiCells;
      return std::make_pair(ie, ip);
    };
    auto keyOf = [&](long ie, long ip) { return ie * 1000003L + ip; };
    std::unordered_map<long, std::vector<unsigned>> grid;
    const double d2max = dupMaxDist_ * dupMaxDist_;
    for (unsigned i : order) {
      const SeedCand& c = cands[i];
      const auto cell = cellOf(c);
      bool dup = false;
      for (long de = -1; de <= 1 && !dup; ++de)
        for (long dp = -1; dp <= 1 && !dup; ++dp) {
          auto it = grid.find(keyOf(cell.first + de, ((cell.second + dp) % nPhiCells + nPhiCells) % nPhiCells));
          if (it == grid.end())
            continue;
          for (unsigned j : it->second) {
            const SeedCand& k = cands[j];
            if (c.charge == k.charge && std::abs(c.eta - k.eta) < dupMaxDEta_ &&
                std::abs(reco::deltaPhi(c.phi, k.phi)) < dupMaxDPhi_ &&
                (c.point - k.point).mag2() < d2max) {
              dup = true;
              break;
            }
          }
        }
      if (dup)
        continue;
      kept.push_back(i);
      grid[keyOf(cell.first, cell.second)].push_back(i);
    }
    std::sort(kept.begin(), kept.end());  // keep the original (jet, window, pixel) order in the output
  } else {
    kept = order;
  }

  result->reserve(kept.size());
  resultTracks->reserve(kept.size());
  for (unsigned i : kept) {
    const SeedCand& c = cands[i];
    edm::OwnVector<TrackingRecHit> hits;
    if (c.hit)
      hits.push_back(c.hit->clone());
    result->emplace_back(c.state, std::move(hits), PropagationDirection::alongMomentum);
    reco::Track::CovarianceMatrix mm;
    resultTracks->emplace_back(1,
                               1,
                               reco::Track::Point(c.point.x(), c.point.y(), c.point.z()),
                               reco::Track::Vector(c.dir.x(), c.dir.y(), c.dir.z()),
                               1,
                               mm);
  }
  LogDebug("DeepCoreSeedGeneratorPhase2") << "event " << iEvent.id() << ": " << cands.size() << " seeds before, "
                                          << result->size() << " after duplicate removal";
  iEvent.put(std::move(result));
  iEvent.put(std::move(resultTracks));
}

// =================================================================================================
// helpers (same as DeepCoreNtuplizerPhase2)
// =================================================================================================

std::pair<bool, Basic3DVector<float>> DeepCoreSeedGeneratorPhase2::findIntersection(
    const GlobalVector& dir, const reco::Candidate::Point& vertex, const GeomDet* det) const {
  if (!det)
    return {false, Basic3DVector<float>(0, 0, 0)};
  StraightLinePlaneCrossing vertexPlane(Basic3DVector<float>(vertex.x(), vertex.y(), vertex.z()),
                                        Basic3DVector<float>(dir.x(), dir.y(), dir.z()));
  // only crossings in front of the vertex (the full line also crosses the plane on the other side)
  const auto s = vertexPlane.pathLength(det->specificSurface());
  if (!s.first || s.second <= 0)
    return {false, Basic3DVector<float>(0, 0, 0)};
  return {true, vertexPlane.position(s.second)};
}

bool DeepCoreSeedGeneratorPhase2::pitchOK(const GeomDet* det) const {
  const auto* pdu = dynamic_cast<const PixelGeomDetUnit*>(det);
  if (!pdu)
    return false;
  const auto p = pdu->specificTopology().pitch();
  return std::abs(p.first - pitchX_) < 0.01 * pitchX_ && std::abs(p.second - pitchY_) < 0.01 * pitchY_;
}

std::pair<int, int> DeepCoreSeedGeneratorPhase2::local2Pixel(double locX, double locY, const GeomDet* det) const {
  const LocalPoint locXY(locX, locY);
  const auto pix = static_cast<const PixelGeomDetUnit*>(det)->specificTopology().pixel(locXY);
  // same truncation (float -> int) as the ntuplizer
  return {(int)pix.first, (int)pix.second};
}

int DeepCoreSeedGeneratorPhase2::pixelFlipper(const GeomDet* det) const {
  const GlobalVector globZdir = det->specificSurface().toGlobal(LocalVector(0, 0, 1));
  const GlobalPoint& c = det->position();
  const float direction = globZdir.x() * c.x() + globZdir.y() * c.y() + globZdir.z() * c.z();
  return direction < 0 ? -1 : 1;
}

void DeepCoreSeedGeneratorPhase2::fillPixelMatrix(
    const SiPixelCluster& cluster, int layer, const LocalPoint& inter, const GeomDet* det, float* image) const {
  if (layer < 1 || layer > Nlayer)
    return;
  const int flip = pixelFlipper(det);
  const std::pair<int, int> pixInter = local2Pixel(inter.x(), inter.y(), det);
  for (int i = 0; i < cluster.size(); ++i) {
    const SiPixelCluster::Pixel pix = cluster.pixel(i);
    int nx = flip * (pix.x - pixInter.first);
    int ny = pix.y - pixInter.second;
    if (std::abs(nx) < jetDimX / 2 && std::abs(ny) < jetDimY / 2) {
      nx += jetDimX / 2;
      ny += jetDimY / 2;
      if (nx >= 0 && nx < jetDimX && ny >= 0 && ny < jetDimY)
        image[(nx * jetDimY + ny) * Nlayer + (layer - 1)] += pix.adc / chargeNorm_;
    }
  }
}

const GeomDet* DeepCoreSeedGeneratorPhase2::detectorSelector(int llay,
                                                             const GlobalVector& dir,
                                                             const reco::Vertex& jetVertex,
                                                             const TrackerTopology* tTopo,
                                                             const edmNew::DetSetVector<SiPixelCluster>& clusters) const {
  // barrel module of layer llay whose plane is crossed closest to its centre. The ntuplizer considers the
  // modules with SimHits; here the modules with clusters are used.
  const GeomDet* best = nullptr;
  double bestDist2 = 0.;
  const reco::Candidate::Point vtx(jetVertex.position());
  for (const auto& detset : clusters) {
    const DetId id(detset.id());
    if (id.subdetId() != PixelSubdetector::PixelBarrel)
      continue;
    if ((int)tTopo->layer(id) != llay)
      continue;
    const GeomDet* det = geometry_->idToDet(id);
    auto interPair = findIntersection(dir, vtx, det);
    if (!interPair.first)
      continue;
    const LocalPoint li = det->specificSurface().toLocal(GlobalPoint(interPair.second));
    const double d2 = li.x() * li.x() + li.y() * li.y();
    if (best == nullptr || d2 < bestDist2) {
      best = det;
      bestDist2 = d2;
    }
  }
  return best;
}

std::vector<GlobalVector> DeepCoreSeedGeneratorPhase2::splittedClusterDirections(
    const reco::Candidate& jet,
    const TrackerTopology* tTopo,
    const PixelClusterParameterEstimator* pixelCPE,
    const reco::Vertex& jetVertex,
    int layer,
    const edmNew::DetSetVector<SiPixelCluster>& clusters) const {
  std::vector<GlobalVector> clustDirs;
  const GlobalVector jetDir(jet.px(), jet.py(), jet.pz());
  const GlobalPoint ppv(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
  for (const auto& detset : clusters) {
    const DetId id(detset.id());
    const bool isEndCap = (id.subdetId() == PixelSubdetector::PixelEndcap);
    int lay = tTopo->layer(id);
    if (isEndCap)
      lay += 4;  // same layer counting as the ntuplizer
    if (lay != layer)
      continue;
    const GeomDet* det = geometry_->idToDet(id);
    const GeomDetUnit* detUnit = geometry_->idToDetUnit(id);
    const LocalVector jetDirLocal = det->surface().toLocal(jetDir);
    const float jetTanAlpha = jetDirLocal.x() / jetDirLocal.z();
    const float jetTanBeta = jetDirLocal.y() / jetDirLocal.z();
    const float jetZOverRho = std::sqrt(jetTanAlpha * jetTanAlpha + jetTanBeta * jetTanBeta);
    const auto pitch = static_cast<const PixelTopology&>(det->topology()).pitch();
    const float thickness = det->surface().bounds().thickness();
    float tanLA = tanLorentzAngle_;
    if (!isEndCap && tTopo->pxbLayer(id) == 1)
      tanLA = tanLorentzAngleBarrelLayer1_;
    float expSizeX = expSizeXAtLorentzAngleIncidence_ + std::abs(expSizeXDeltaPerTanAlpha_ * (jetTanAlpha - tanLA));
    float expSizeY = std::sqrt(expSizeYAtNormalIncidence_ * expSizeYAtNormalIncidence_ +
                               thickness * thickness / (pitch.second * pitch.second) * jetTanBeta * jetTanBeta);
    expSizeX = std::max(expSizeX, 1.f);
    expSizeY = std::max(expSizeY, 1.f);
    const float expCharge = std::sqrt(1.08f + jetZOverRho * jetZOverRho) * centralMIPCharge_;

    for (const auto& aCluster : detset) {
      const GlobalPoint cPos = det->surface().toGlobal(pixelCPE->localParametersV(aCluster, *detUnit)[0].first);
      const GlobalVector clusterDir = cPos - ppv;
      if (Geom::deltaR(jetDir, clusterDir) > deltaR_)
        continue;
      if (aCluster.charge() > expCharge * chargeFracMin_ &&
          (aCluster.sizeX() > expSizeX + 1 || aCluster.sizeY() > expSizeY + 1))
        clustDirs.push_back(clusterDir);
    }
  }
  return clustDirs;
}

void DeepCoreSeedGeneratorPhase2::dumpWindow(const edm::EventID& eid,
                                             int jetIndex,
                                             const reco::Candidate& jet,
                                             const Window& w,
                                             const float* image,
                                             const float* prob,
                                             const float* par) const {
  // binary dump for DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/test/compare_images.py:
  //   8 doubles (run, event, jet index, direction index, jet eta, jet pt, X, Y), then image [X][Y][L],
  //   prob [X][Y][Nover] (slot 0 of the probability output) and par [X][Y][Nover][Npar+1], as float32
  std::ostringstream fn;
  fn << dumpDir_ << "/r" << eid.run() << "_e" << eid.event() << "_j" << jetIndex << "_d" << w.index << ".bin";
  std::ofstream f(fn.str(), std::ios::binary);
  const double head[8] = {(double)eid.run(), (double)eid.event(), (double)jetIndex, (double)w.index,
                          jet.eta(),         jet.pt(),            (double)jetDimX,  (double)jetDimY};
  f.write(reinterpret_cast<const char*>(head), sizeof(head));
  f.write(reinterpret_cast<const char*>(image), sizeof(float) * imageSize);
  for (int c = 0; c < jetDimX * jetDimY * Nover; ++c)
    f.write(reinterpret_cast<const char*>(prob + c * NprobOut), sizeof(float));
  f.write(reinterpret_cast<const char*>(par), sizeof(float) * jetDimX * jetDimY * Nover * NparOut);
}

// =================================================================================================

void DeepCoreSeedGeneratorPhase2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("pixelClusters", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<edm::InputTag>("cores", edm::InputTag("ak4CaloJets"));
  desc.add<double>("ptMin", 500.)->setComment("jet pT threshold; the barrel training used 500 GeV");
  desc.add<double>("absEtaMax", 1.4)->setComment("the network is trained on barrel jets only (|eta| < 1.4)");
  desc.add<double>("deltaR", 0.1);
  // merged-cluster test, Phase-2 values (as in DeepCoreNtuplizerPhase2 / phase2_DeepCoreNtuplizer.py)
  desc.add<double>("centralMIPCharge", 12000.0);
  desc.add<double>("chargeFractionMin", 2.0);
  desc.add<double>("expSizeXAtLorentzAngleIncidence", 1.5);
  desc.add<double>("expSizeXDeltaPerTanAlpha", 5.2);
  desc.add<double>("expSizeYAtNormalIncidence", 2.3);
  desc.add<double>("tanLorentzAngle", -0.21);
  desc.add<double>("tanLorentzAngleBarrelLayer1", 0.0);
  desc.add<double>("chargeNorm", 14000.)->setComment("must equal the ntuplizer chargeNorm used for the training");
  desc.add<double>("pitchX", 0.0025);
  desc.add<double>("pitchY", 0.010);
  desc.add<std::string>("pixelCPE", "PixelCPEGeneric");
  desc.add<edm::FileInPath>("weightFile",
                            edm::FileInPath("DeepCoreSeeding/DeepCoreSeedGeneratorPhase2/data/DeepCore_phase2_ep088.onnx"));
  desc.add<std::vector<std::string>>("inputTensorName", {"input_1", "input_2", "input_3"});
  desc.add<std::vector<std::string>>("outputTensorName", {"output_node0", "output_node1"});
  desc.add<double>("probThr", 0.85);
  desc.add<std::vector<double>>("probThrDeltaOverlap", {0., 0., 0.})
      ->setComment("added to probThr for overlap slot 0,1,2 (DeepCore 2.2.1 Phase-1: 0, 0.15, 0.3)");
  desc.add<std::vector<double>>("probThrDeltaL134off", {0., 0., 0.})
      ->setComment("added when layers 1, 3 and 4 are empty (DeepCore 2.2.1 Phase-1: 0.1, 0.05, 0)");
  desc.add<double>("probThrDeltaL2off", 0.)->setComment("added when layer 2 is empty (DeepCore 2.2.1 Phase-1: 0.3)");
  desc.add<double>("seedPtMin", 1.0)->setComment("lower bound on the predicted pT [GeV]");
  desc.add<std::vector<double>>("seedErrors", {0.15 * 0.15, 0.5e-5, 0.5e-5, 2e-5, 2e-5})
      ->setComment("diagonal of the seed covariance: (q/p, dxdz, dydz, x [cm^2], y [cm^2])");
  desc.add<unsigned int>("maxBatch", 16);
  desc.add<double>("duplicateMaxDist", 0.)
      ->setComment("seed duplicate removal: two seeds are duplicates if their points on the layer-2 module are closer "
                   "than this [cm] AND their directions differ by less than duplicateMaxDEta and duplicateMaxDPhi; "
                   "the one with the higher probability is kept. 0 = off (only exact duplicates are removed)");
  desc.add<double>("duplicateMaxDEta", 0.002);
  desc.add<int>("seedCharge", 1)->setComment("+1 (default, as in Phase-1), -1, or 0 = one seed with each charge");
  desc.add<double>("seedQoverPRelError", 0.)
      ->setComment("if > 0: sigma(q/p) = this x |q/p| of the predicted momentum, instead of seedErrors[0]");
  desc.add<bool>("attachHit", false)
      ->setComment("attach the nearest attachHitClusters cluster on the layer-2 module to the seed and start "
                   "from its position (the seed is then subject to the RedundantSeedCleaner of the candidate maker)");
  desc.add<edm::InputTag>("attachHitClusters", edm::InputTag("siPixelClusters"));
  desc.add<double>("attachMaxDX", 0.01)->setComment("[cm] max |dx| between prediction and cluster (4 pitches)");
  desc.add<double>("attachMaxDY", 0.03)->setComment("[cm] max |dy| between prediction and cluster (3 pitches)");
  desc.add<bool>("attachHitRequired", false)->setComment("drop predictions without a cluster nearby");
  desc.add<double>("duplicateMaxDPhi", 0.002);
  desc.addUntracked<std::string>("dumpDir", "")
      ->setComment("if set, write every window (image + outputs) there, for test/compare_images.py");
  descriptions.add("deepCoreSeedGeneratorPhase2", desc);
}

DEFINE_FWK_MODULE(DeepCoreSeedGeneratorPhase2);
