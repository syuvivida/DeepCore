// DeepCoreSeedGeneratorONNX: the Phase-1 DeepCore seed generator of CMSSW
// (RecoTracker/TkSeedGenerator/plugins/DeepCoreSeedGenerator.cc, CMSSW_14_2_2) with the TensorFlow
// inference replaced by ONNX Runtime, to run a PyTorch-trained model exported to ONNX.
// Everything else (windows, image filling, thresholds, seed parameters and errors) is unchanged.
// -*- C++ -*-
//
// Package:    trackJet/DeepCoreSeedGeneratorONNX
// Class:      DeepCoreSeedGeneratorONNX
//
/**\class DeepCoreSeedGeneratorONNX DeepCoreSeedGeneratorONNX.cc trackJet/DeepCoreSeedGeneratorONNX/plugins/DeepCoreSeedGeneratorONNX.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Valerio Bertacchi
//         Created:  Mon, 18 Dec 2017 16:35:04 GMT
//
//

// system include files

#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

using cms::Ort::ONNXRuntime;

//
// class declaration
//

// Network input of one window: jet eta, jet pT and the 30x30x4 pixel image (NHWC, as the TensorFlow model)
struct DeepCoreInput {
  float eta = 0.f;
  float pt = 0.f;
  std::vector<float> image;
};

class DeepCoreSeedGeneratorONNX : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit DeepCoreSeedGeneratorONNX(const edm::ParameterSet&, const ONNXRuntime*);
  ~DeepCoreSeedGeneratorONNX() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  // A pointer to a cluster and a list of tracks on it

  // static methods for handling the global cache
  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

  double jetPt_;
  double jetEta_;
  double pitchX_ = 0.01;              //100 um (pixel pitch in X)
  double pitchY_ = 0.015;             //150 um (pixel pitch in Y)
  static constexpr int jetDimX = 30;  //pixel dimension of NN window on layer2
  static constexpr int jetDimY = 30;  //pixel dimension of NN window on layer2
  static constexpr int Nlayer = 4;    //Number of layer used in DeepCore
  static constexpr int Nover = 3;     //Max number of tracks recorded per pixel
  static constexpr int Npar = 5;      //Number of track parameter

  // DeepCore 2.2.1 thresholds delta
  double dth[3] = {0.0, 0.15, 0.3};
  double dthl[3] = {0.1, 0.05, 0};

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  std::string propagatorName_;
  const GlobalTrackingGeometry* geometry_ = nullptr;

  edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusters_;
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> inputPixelClusters_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> cores_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geometryToken_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> pixelCPEToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  double ptMin_;
  double deltaR_;
  double chargeFracMin_;
  double centralMIPCharge_;
  std::string weightfilename_;
  std::vector<std::string> inputTensorName_;
  std::vector<std::string> outputTensorName_;
  double probThr_;

  std::pair<bool, Basic3DVector<float>> findIntersection(const GlobalVector&,
                                                         const reco::Candidate::Point&,
                                                         const GeomDet*);

  void fillPixelMatrix(const SiPixelCluster&,
                       int,
                       Point3DBase<float, LocalTag>,
                       const GeomDet*,
                       DeepCoreInput&);

  std::pair<int, int> local2Pixel(double, double, const GeomDet*);

  LocalPoint pixel2Local(int, int, const GeomDet*);

  int pixelFlipper(const GeomDet*);

  const GeomDet* DetectorSelector(int,
                                  const reco::Candidate&,
                                  GlobalVector,
                                  const reco::Vertex&,
                                  const TrackerTopology* const,
                                  const edmNew::DetSetVector<SiPixelCluster>&);

  std::vector<GlobalVector> splittedClusterDirections(
      const reco::Candidate&,
      const TrackerTopology* const,
      const PixelClusterParameterEstimator*,
      const reco::Vertex&,
      int,
      const edmNew::DetSetVector<SiPixelCluster>&);  //if not working,: args=2 auto

  std::pair<double[jetDimX][jetDimY][Nover][Npar], double[jetDimX][jetDimY][Nover]> SeedEvaluation(
      const DeepCoreInput&);
  static size_t imageIndex(int x, int y, int l) { return ((size_t)x * jetDimY + y) * Nlayer + l; }
};

DeepCoreSeedGeneratorONNX::DeepCoreSeedGeneratorONNX(const edm::ParameterSet& iConfig, const ONNXRuntime*)
    : vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      pixelClusters_(
          consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("pixelClusters"))),
      cores_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("cores"))),
      geometryToken_(esConsumes()),
      pixelCPEToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("pixelCPE")))),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      chargeFracMin_(iConfig.getParameter<double>("chargeFractionMin")),
      centralMIPCharge_(iConfig.getParameter<double>("centralMIPCharge")),
      weightfilename_(iConfig.getParameter<edm::FileInPath>("weightFile").fullPath()),
      inputTensorName_(iConfig.getParameter<std::vector<std::string>>("inputTensorName")),
      outputTensorName_(iConfig.getParameter<std::vector<std::string>>("outputTensorName")),
      probThr_(iConfig.getParameter<double>("probThr"))

{
  produces<TrajectorySeedCollection>();
  produces<reco::TrackCollection>();
}

DeepCoreSeedGeneratorONNX::~DeepCoreSeedGeneratorONNX() {}

void DeepCoreSeedGeneratorONNX::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto result = std::make_unique<TrajectorySeedCollection>();
  auto resultTracks = std::make_unique<reco::TrackCollection>();


  using namespace edm;
  using namespace reco;

  geometry_ = &iSetup.getData(geometryToken_);

  const auto& inputPixelClusters_ = iEvent.get(pixelClusters_);
  const auto& vertices = iEvent.get(vertices_);
  const auto& cores = iEvent.get(cores_);

  const PixelClusterParameterEstimator* pixelCPE = &iSetup.getData(pixelCPEToken_);

  const TrackerTopology* const tTopo = &iSetup.getData(topoToken_);
  auto output = std::make_unique<edmNew::DetSetVector<SiPixelCluster>>();

  for (const auto& jet : cores) {  // jet loop

    if (jet.pt() > ptMin_) {
      std::set<unsigned long long int> ids;
      const reco::Vertex& jetVertex = vertices[0];

      std::vector<GlobalVector> splitClustDirSet =
          splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 2, inputPixelClusters_);
      bool l2off = (splitClustDirSet.empty());  // no adc BPIX2
      splitClustDirSet = splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 3, inputPixelClusters_);
      bool l134off = (splitClustDirSet.empty());  // no adc BPIX1 or BPIX3 or BPIX4
      splitClustDirSet = splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 4, inputPixelClusters_);
      l134off = (splitClustDirSet.empty() && l134off);
      splitClustDirSet = splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 1, inputPixelClusters_);
      l134off = (splitClustDirSet.empty() && l134off);

      if (splitClustDirSet.empty()) {  //if layer 1 is broken find direcitons on layer 2
        splitClustDirSet = splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 2, inputPixelClusters_);
      }
      splitClustDirSet.emplace_back(GlobalVector(jet.px(), jet.py(), jet.pz()));
      for (int cc = 0; cc < (int)splitClustDirSet.size(); cc++) {
        DeepCoreInput input_tensors;  // network input, all zero
        input_tensors.image.assign((size_t)jetDimX * jetDimY * Nlayer, 0.f);

        GlobalVector bigClustDir = splitClustDirSet[cc];

        jetEta_ = jet.eta();
        jetPt_ = jet.pt();
        input_tensors.eta = jet.eta();
        input_tensors.pt = jet.pt();

        const GeomDet* globDet = DetectorSelector(
            2, jet, bigClustDir, jetVertex, tTopo, inputPixelClusters_);  //det. aligned to bigClustDir on L2

        if (globDet == nullptr)  //no intersection between bigClustDir and pixel detector modules found
          continue;

        const GeomDet* goodDet1 = DetectorSelector(1, jet, bigClustDir, jetVertex, tTopo, inputPixelClusters_);
        const GeomDet* goodDet3 = DetectorSelector(3, jet, bigClustDir, jetVertex, tTopo, inputPixelClusters_);
        const GeomDet* goodDet4 = DetectorSelector(4, jet, bigClustDir, jetVertex, tTopo, inputPixelClusters_);

        for (const auto& detset : inputPixelClusters_) {
          const GeomDet* det = geometry_->idToDet(detset.id());

          for (const auto& aCluster : detset) {
            det_id_type aClusterID = detset.id();
            if (DetId(aClusterID).subdetId() != PixelSubdetector::PixelBarrel)
              continue;

            int lay = tTopo->layer(det->geographicalId());

            std::pair<bool, Basic3DVector<float>> interPair =
                findIntersection(bigClustDir, (reco::Candidate::Point)jetVertex.position(), det);
            if (interPair.first == false)
              continue;
            Basic3DVector<float> inter = interPair.second;
            auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);

            GlobalPoint pointVertex(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());

            LocalPoint clustPos_local =
                pixelCPE->localParametersV(aCluster, (*geometry_->idToDetUnit(detset.id())))[0].first;

            if (std::abs(clustPos_local.x() - localInter.x()) / pitchX_ <= jetDimX / 2 &&
                std::abs(clustPos_local.y() - localInter.y()) / pitchY_ <=
                    jetDimY / 2) {  //used the baricenter, better description maybe useful

              if (det == goodDet1 || det == goodDet3 || det == goodDet4 || det == globDet) {
                fillPixelMatrix(aCluster, lay, localInter, det, input_tensors);
              }
            }  //cluster in ROI
          }  //cluster
        }  //detset

        //here the NN produce the seed from the filled input
        std::pair<double[jetDimX][jetDimY][Nover][Npar], double[jetDimX][jetDimY][Nover]> seedParamNN =
            DeepCoreSeedGeneratorONNX::SeedEvaluation(input_tensors);

        for (int i = 0; i < jetDimX; i++) {
          for (int j = 0; j < jetDimY; j++) {
            for (int o = 0; o < Nover; o++) {
              if (seedParamNN.second[i][j][o] >
                  (probThr_ + dth[o] + (l2off ? 0.3 : 0) + (l134off ? dthl[o] : 0))) {  // DeepCore 2.2.1 Threshold
                std::pair<bool, Basic3DVector<float>> interPair =
                    findIntersection(bigClustDir, (reco::Candidate::Point)jetVertex.position(), globDet);
                auto localInter = globDet->specificSurface().toLocal((GlobalPoint)interPair.second);

                int flip = pixelFlipper(globDet);  // 1=not flip, -1=flip
                int nx = i - jetDimX / 2;
                int ny = j - jetDimY / 2;
                nx = flip * nx;
                std::pair<int, int> pixInter = local2Pixel(localInter.x(), localInter.y(), globDet);
                nx = nx + pixInter.first;
                ny = ny + pixInter.second;
                LocalPoint xyLocal = pixel2Local(nx, ny, globDet);

                double xx = xyLocal.x() + seedParamNN.first[i][j][o][0] * 0.01;  //0.01=internal normalization of NN
                double yy = xyLocal.y() + seedParamNN.first[i][j][o][1] * 0.01;  //0.01=internal normalization of NN
                LocalPoint localSeedPoint = LocalPoint(xx, yy, 0);

                double track_eta =
                    seedParamNN.first[i][j][o][2] * 0.01 + bigClustDir.eta();  //pay attention to this 0.01
                double track_theta = 2 * std::atan(std::exp(-track_eta));
                double track_phi =
                    seedParamNN.first[i][j][o][3] * 0.01 + bigClustDir.phi();  //pay attention to this 0.01
                double pt = seedParamNN.first[i][j][o][4];
                double normdirR = pt / sin(track_theta);

                const GlobalVector globSeedDir(
                    GlobalVector::Polar(Geom::Theta<double>(track_theta), Geom::Phi<double>(track_phi), normdirR));
                LocalVector localSeedDir = globDet->surface().toLocal(globSeedDir);
                uint64_t seedid = (uint64_t(xx * 200.) << 0) + (uint64_t(yy * 200.) << 16) +
                                  (uint64_t(track_eta * 400.) << 32) + (uint64_t(track_phi * 400.) << 48);
                if (ids.count(seedid) != 0) {
                  continue;
                }
                //to disable jetcore iteration comment from here to probability block end--->
                //seeing iteration skipped, useful to total eff and FakeRate comparison
                ids.insert(seedid);

                //Covariance matrix, the hadrcoded variances = NN residuals width
                //(see https://twiki.cern.ch/twiki/bin/view/CMSPublic/NNJetCoreAtCtD2019)
                float em[15] = {0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0};   // (see LocalTrajectoryError for details), order as follow:
                em[0] = 0.15 * 0.15;  // q/pt
                em[2] = 0.5e-5;       // dxdz
                em[5] = 0.5e-5;       // dydz
                em[9] = 2e-5;         // x
                em[14] = 2e-5;        // y
                long int detId = globDet->geographicalId();
                LocalTrajectoryParameters localParam(localSeedPoint, localSeedDir, TrackCharge(1));
                result->emplace_back(TrajectorySeed(PTrajectoryStateOnDet(localParam, pt, em, detId, /*surfaceSide*/ 0),
                                                    edm::OwnVector<TrackingRecHit>(),
                                                    PropagationDirection::alongMomentum));

                GlobalPoint globalSeedPoint = globDet->surface().toGlobal(localSeedPoint);
                reco::Track::CovarianceMatrix mm;
                resultTracks->emplace_back(
                    reco::Track(1,
                                1,
                                reco::Track::Point(globalSeedPoint.x(), globalSeedPoint.y(), globalSeedPoint.z()),
                                reco::Track::Vector(globSeedDir.x(), globSeedDir.y(), globSeedDir.z()),
                                1,
                                mm));
              }  //probability
            }
          }
        }
      }  //bigcluster
    }  //jet > pt
  }  //jet
  iEvent.put(std::move(result));
  iEvent.put(std::move(resultTracks));
}

std::pair<bool, Basic3DVector<float>> DeepCoreSeedGeneratorONNX::findIntersection(const GlobalVector& dir,
                                                                              const reco::Candidate::Point& vertex,
                                                                              const GeomDet* det) {
  StraightLinePlaneCrossing vertexPlane(Basic3DVector<float>(vertex.x(), vertex.y(), vertex.z()),
                                        Basic3DVector<float>(dir.x(), dir.y(), dir.z()));

  std::pair<bool, Basic3DVector<float>> pos = vertexPlane.position(det->specificSurface());

  return pos;
}

std::pair<int, int> DeepCoreSeedGeneratorONNX::local2Pixel(double locX, double locY, const GeomDet* det) {
  LocalPoint locXY(locX, locY);
  float pixX = (dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).first;
  float pixY = (dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).second;
  std::pair<int, int> out(pixX, pixY);
  return out;
}

LocalPoint DeepCoreSeedGeneratorONNX::pixel2Local(int pixX, int pixY, const GeomDet* det) {
  float locX = (dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localX(pixX);
  float locY = (dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localY(pixY);
  LocalPoint locXY(locX, locY);
  return locXY;
}

int DeepCoreSeedGeneratorONNX::pixelFlipper(const GeomDet* det) {
  int out = 1;
  LocalVector locZdir(0, 0, 1);
  GlobalVector globZdir = det->specificSurface().toGlobal(locZdir);
  const GlobalPoint& globDetCenter = det->position();
  float direction =
      globZdir.x() * globDetCenter.x() + globZdir.y() * globDetCenter.y() + globZdir.z() * globDetCenter.z();
  if (direction < 0)
    out = -1;
  return out;
}

void DeepCoreSeedGeneratorONNX::fillPixelMatrix(const SiPixelCluster& cluster,
                                            int layer,
                                            Point3DBase<float, LocalTag> inter,
                                            const GeomDet* det,
                                            DeepCoreInput& input_tensors) {
  int flip = pixelFlipper(det);  // 1=not flip, -1=flip

  for (int i = 0; i < cluster.size(); i++) {
    SiPixelCluster::Pixel pix = cluster.pixel(i);
    std::pair<int, int> pixInter = local2Pixel(inter.x(), inter.y(), det);
    int nx = pix.x - pixInter.first;
    int ny = pix.y - pixInter.second;
    nx = flip * nx;

    if (abs(nx) < jetDimX / 2 && abs(ny) < jetDimY / 2) {
      nx = nx + jetDimX / 2;
      ny = ny + jetDimY / 2;
      //14000 = normalization of ACD counts used in the NN
      input_tensors.image[imageIndex(nx, ny, layer - 1)] += (pix.adc) / (14000.f);
    }
  }
}

std::pair<double[DeepCoreSeedGeneratorONNX::jetDimX][DeepCoreSeedGeneratorONNX::jetDimY][DeepCoreSeedGeneratorONNX::Nover]
                [DeepCoreSeedGeneratorONNX::Npar],
          double[DeepCoreSeedGeneratorONNX::jetDimX][DeepCoreSeedGeneratorONNX::jetDimY][DeepCoreSeedGeneratorONNX::Nover]>
DeepCoreSeedGeneratorONNX::SeedEvaluation(const DeepCoreInput& in) {
  // same inputs and outputs as the TensorFlow model, through ONNX Runtime:
  //   inputs  (eta [1,1], pt [1,1], image [1,30,30,4]),  outputs (par [1,30,30,3,nPar], prob [1,30,30,3,nProb])
  // nPar and nProb are read from the output sizes (the PyTorch model also returns a "zero flag" column)
  cms::Ort::FloatArrays input(3);
  input[0] = {in.eta};
  input[1] = {in.pt};
  input[2] = in.image;
  const std::vector<std::vector<int64_t>> shapes = {{1, 1}, {1, 1}, {1, jetDimX, jetDimY, Nlayer}};
  cms::Ort::FloatArrays out = globalCache()->run(inputTensorName_, input, shapes, outputTensorName_, 1);
  const size_t cells = (size_t)jetDimX * jetDimY * Nover;
  if (out.size() != 2 || out[0].size() % cells != 0 || out[1].size() % cells != 0 ||
      out[0].size() / cells < (size_t)Npar || out[1].size() / cells < 1)
    throw cms::Exception("DeepCoreSeedGeneratorONNX")
        << "unexpected ONNX output sizes " << (out.size() > 0 ? out[0].size() : 0) << ", "
        << (out.size() > 1 ? out[1].size() : 0) << " for a " << jetDimX << "x" << jetDimY << "x" << Nover
        << " output (outputTensorName must list the parameter output first, then the probability)";
  const size_t nPar = out[0].size() / cells, nProb = out[1].size() / cells;

  std::pair<double[jetDimX][jetDimY][Nover][Npar], double[jetDimX][jetDimY][Nover]> output_combined;
  for (int x = 0; x < jetDimX; x++) {
    for (int y = 0; y < jetDimY; y++) {
      for (int trk = 0; trk < Nover; trk++) {
        const size_t cell = ((size_t)x * jetDimY + y) * Nover + trk;
        output_combined.second[x][y][trk] = out[1][cell * nProb];
        for (int p = 0; p < Npar; p++)
          output_combined.first[x][y][trk][p] = out[0][cell * nPar + p];
      }
    }
  }
  return output_combined;
}

const GeomDet* DeepCoreSeedGeneratorONNX::DetectorSelector(int llay,
                                                       const reco::Candidate& jet,
                                                       GlobalVector jetDir,
                                                       const reco::Vertex& jetVertex,
                                                       const TrackerTopology* const tTopo,
                                                       const edmNew::DetSetVector<SiPixelCluster>& clusters) {
  double minDist = 0.0;
  GeomDet* output = (GeomDet*)nullptr;
  for (const auto& detset : clusters) {
    auto aClusterID = detset.id();
    if (DetId(aClusterID).subdetId() != 1)
      continue;
    const GeomDet* det = geometry_->idToDet(aClusterID);
    int lay = tTopo->layer(det->geographicalId());
    if (lay != llay)
      continue;
    std::pair<bool, Basic3DVector<float>> interPair =
        findIntersection(jetDir, (reco::Candidate::Point)jetVertex.position(), det);
    if (interPair.first == false)
      continue;
    Basic3DVector<float> inter = interPair.second;
    auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);
    if ((minDist == 0.0 || std::abs(localInter.x()) < minDist) && std::abs(localInter.y()) < 3.35) {
      minDist = std::abs(localInter.x());
      output = (GeomDet*)det;
    }
  }  //detset
  return output;
}

std::vector<GlobalVector> DeepCoreSeedGeneratorONNX::splittedClusterDirections(
    const reco::Candidate& jet,
    const TrackerTopology* const tTopo,
    const PixelClusterParameterEstimator* pixelCPE,
    const reco::Vertex& jetVertex,
    int layer,
    const edmNew::DetSetVector<SiPixelCluster>& clusters) {
  std::vector<GlobalVector> clustDirs;
  for (const auto& detset_int : clusters) {
    const GeomDet* det_int = geometry_->idToDet(detset_int.id());
    int lay = tTopo->layer(det_int->geographicalId());
    if (lay != layer)
      continue;  //NB: saved bigClusters on all the layers!!
    auto detUnit = geometry_->idToDetUnit(detset_int.id());
    for (const auto& aCluster : detset_int) {
      GlobalPoint clustPos = det_int->surface().toGlobal(pixelCPE->localParametersV(aCluster, (*detUnit))[0].first);
      GlobalPoint vertexPos(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
      GlobalVector clusterDir = clustPos - vertexPos;
      GlobalVector jetDir(jet.px(), jet.py(), jet.pz());
      if (Geom::deltaR(jetDir, clusterDir) < deltaR_) {
        clustDirs.emplace_back(clusterDir);
      }
    }
  }
  return clustDirs;
}

std::unique_ptr<ONNXRuntime> DeepCoreSeedGeneratorONNX::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("weightFile").fullPath());
}

void DeepCoreSeedGeneratorONNX::globalEndJob(const ONNXRuntime*) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DeepCoreSeedGeneratorONNX::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("pixelClusters", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<edm::InputTag>("cores", edm::InputTag("jetsForCoreTracking"));
  desc.add<double>("ptMin", 100);    // the current training uses 500 GeV
  desc.add<double>("deltaR", 0.25);  // the current training uses 0.1
  desc.add<double>("chargeFractionMin", 18000.0);
  desc.add<double>("centralMIPCharge", 2);
  desc.add<std::string>("pixelCPE", "PixelCPEGeneric");
  desc.add<edm::FileInPath>(
      "weightFile",
      edm::FileInPath("DeepCoreSeeding/DeepCoreSeedGeneratorONNXPhase1/data/DeepCore_phase1.onnx"));
  desc.add<std::vector<std::string>>("inputTensorName", {"input_1", "input_2", "input_3"});
  desc.add<std::vector<std::string>>("outputTensorName", {"output_node0", "output_node1"});
  desc.add<double>("probThr", 0.7);  // DeepCore 2.2.1 baseline threshold
  descriptions.add("deepCoreSeedGeneratorONNX", desc);
}

DEFINE_FWK_MODULE(DeepCoreSeedGeneratorONNX);
