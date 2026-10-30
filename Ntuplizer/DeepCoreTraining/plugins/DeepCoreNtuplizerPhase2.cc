// -*- C++ -*-
//
// Package:    RecoTracker/DeepCoreTraining
// Class:      DeepCoreNtuplizerPhase2
//
/**\class DeepCoreNtuplizerPhase2 DeepCoreNtuplizerPhase2.cc RecoTracker/DeepCoreTraining/plugins/DeepCoreNtuplizerPhase2.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Valerio Bertacchi
//         Created:  Fri, 09 Jul 2021 10:43:34 GMT
// v2             : Hichem Bouchamaoui
// Updated:       Shin-Shan Eiko Yu
// Phase-2 Inner Tracker version:
//   - window 120 x 45 pixels of 25 x 100 um = same physical size (3.0 x 4.5 mm) as the Phase-1 30 x 30 window;
//     modules whose topology pitch is not 25 x 100 um are skipped
//   - merged-cluster test as in JetCoreClusterSplitter, thickness/pitch from the geometry, Phase-2 parameters
//   - forward-only line/plane intersection (no module on the opposite side of the detector)
//   - pixel centres from PixelTopology::localX/Y(i+0.5) (chip-edge pixels handled by the topology)
//   - NPixLay bound check (endcap "layers" 5-16 in Phase-2)
//   - SimTrack/PSimHit matching also on eventId and particle type
//   - configurable image charge normalization (chargeNorm); g4SimHits module label in the SimHit defaults
//   - unused 17 MB member clusterSplit removed
//

// system include files
#include <memory>
#include <algorithm>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"
//#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h" for 14_X
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"



#include <boost/range.hpp>
#include <boost/foreach.hpp>
#include "boost/multi_array.hpp"



#include "SimDataFormats/Track/interface/SimTrack.h"

// #include "SimDataFormats/Vertex/interface/SimVertex.h"


//#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h" //for 14_2_X
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"


#include "TTree.h"


//
// class declaration
//

class DeepCoreNtuplizerPhase2 : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit DeepCoreNtuplizerPhase2(const edm::ParameterSet&);
  ~DeepCoreNtuplizerPhase2();


  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  struct TrackAndState
  {
    TrackAndState(const reco::Track *aTrack, TrajectoryStateOnSurface aState) :
      track(aTrack), state(aState) {}
    const reco::Track*      track;
    TrajectoryStateOnSurface state;
  };
  
  template<typename Cluster>
  struct ClusterWithTracks
  {
    ClusterWithTracks(const Cluster &c) : cluster(&c) {}
    const Cluster* cluster;
    std::vector<TrackAndState> tracks;
  };
  
  typedef ClusterWithTracks<SiPixelCluster> SiPixelClusterWithTracks;

  typedef boost::sub_range<std::vector<SiPixelClusterWithTracks> > SiPixelClustersWithTracks;

  TFile* DeepCoreNtuplizerPhase2_out;
  TTree* DeepCoreNtuplizerPhase2Tree;
  // Phase-2 Inner Tracker: 25 um (local x) x 100 um (local y) pixels.
  // The window keeps the physical size of the Phase-1 30x30 window (100x150 um pixels):
  //   x: 30 * 100 um = 3.0 mm = 120 * 25 um,   y: 30 * 150 um = 4.5 mm = 45 * 100 um
  static const int jetDimX = 120;
  static const int jetDimY = 45;
  static const uint Nlayer = 4;  // TBPX layers 1-4 (barrel only, see endcapTrain below)
  static const uint Ntrack = 100;
  static const uint Npar = 5; //added 1/pt
  static const uint Nover = 3; // number of overlapped tracks
  double clusterMeas[jetDimX][jetDimY][Nlayer];
  double trackPar[jetDimX][jetDimY][Nover][Npar+1]; //NOFLAG
  double trackProb[jetDimX][jetDimY][Nover];
  // (Phase-1 member clusterSplit[Ntrack][Nlayer][jetDimX][jetDimY] removed: it was never used and would be
  //  100*4*120*45 doubles = 17 MB with the Phase-2 window)
  double track_pt[Ntrack] = {0.0};
  double track_pz[Ntrack] = {0.0};
  double track_eta[Ntrack] = {0.0};
  double track_phi[Ntrack] = {0.0};
  double jet_pt;
  double jet_p;
  double jet_eta;
  double jet_phi;
  double jet_Ntrack = 0;
  int NPixLay[Nlayer] = {0};
  double absEtaMin = 0.;
  double absEtaMax= 1000.;


  int eventID;


  // nominal Phase-2 pitch; checked against each module's PixelTopology (modules with another pitch are skipped)
  const double pitchX = 0.0025; // 25 um
  const double pitchY = 0.010;  // 100 um
  const int distThr = 2;//4
  bool pitchOK(const GeomDet*);
  bool warnedPitch_ = false;



private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  
  std::string propagatorName_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geomToken;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propToken;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> pixelClusterParameterEstimatorToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopToken;

  edm::ESHandle<MagneticField>          magfield_;
  edm::ESHandle<GlobalTrackingGeometry> geometry_;
  edm::ESHandle<Propagator>             propagator_;


  edm::EDGetTokenT<std::vector<reco::Vertex> > vertices_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
  std::vector<SiPixelClusterWithTracks> allSiPixelClusters;
  std::map<uint32_t, SiPixelClustersWithTracks> siPixelDetsWithClusters;
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClusters;
  edm::EDGetTokenT<edm::View<reco::Candidate> > cores_;
  edm::EDGetTokenT<std::vector<SimTrack> > simtracksToken;
  edm::EDGetTokenT<std::vector<PSimHit> > PSimHitToken;
  edm::Handle<std::vector<PSimHit> > simhits;
  edm::EDGetTokenT<std::vector<PSimHit> > PSimHitECToken;
  edm::Handle<std::vector<PSimHit> > simhitsEC;

    
  double ptMin_;
  double pMin_;
  double deltaR_;
  double chargeFracMin_;
  double centralMIPCharge_;
  // merged-cluster test, same parametrization as JetCoreClusterSplitter (Phase-2 values by default)
  double expSizeXAtLorentzAngleIncidence_;
  double expSizeXDeltaPerTanAlpha_;
  double expSizeYAtNormalIncidence_;
  double tanLorentzAngle_;
  double tanLorentzAngleBarrelLayer1_;
  // normalization of the pixel charge in the image (must match the value used in the seed generator)
  double chargeNorm_;

  
  bool barrelTrain_;
  bool endcapTrain_;




  std::pair<bool, Basic3DVector<float>> findIntersection(const GlobalVector & , const reco::Candidate::Point & ,const GeomDet*);

  void fillPixelMatrix(const SiPixelCluster &, uint, auto, const GeomDet*);

  std::pair<int,int> local2Pixel(double, double, const GeomDet*);
  
  LocalPoint pixel2Local(int, int, const GeomDet*);

  // Pixel size fix:
  std::pair<bool,bool> pixel2Size(int, int, const GeomDet*);
  
  int pixelFlipper(const GeomDet*);

  void fillTrackInfo(const reco::Candidate&, const auto &, auto, auto, auto, const GeomDet*, const PixelClusterParameterEstimator*, std::vector<PSimHit>, const TrackerTopology* const);

  const GeomDet* DetectorSelector(int ,const reco::Candidate& jet, GlobalVector,  const reco::Vertex& jetVertex, const TrackerTopology* const, const PixelClusterParameterEstimator*, const auto &);

  std::vector<GlobalVector> splittedClusterDirections(const reco::Candidate&, const TrackerTopology* const, auto pp, const reco::Vertex& jetVertex, int );


};

DeepCoreNtuplizerPhase2::DeepCoreNtuplizerPhase2(const edm::ParameterSet& iConfig) :
      theFieldToken(esConsumes()),
      geomToken(esConsumes()),
      propToken(esConsumes(edm::ESInputTag("", "AnalyticalPropagator"))),
      pixelClusterParameterEstimatorToken(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("pixelCPE")))),
      tTopToken(esConsumes()),
      vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      pixelClusters_(consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"))),
      cores_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("cores"))),
      simtracksToken(consumes<std::vector<SimTrack> >(iConfig.getParameter<edm::InputTag>("simTracks"))),
      PSimHitToken(consumes<std::vector<PSimHit> >(iConfig.getParameter<edm::InputTag>("simHit"))),
      PSimHitECToken(consumes<std::vector<PSimHit> >(iConfig.getParameter<edm::InputTag>("simHitEC"))),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      pMin_(iConfig.getParameter<double>("pMin")),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      chargeFracMin_(iConfig.getParameter<double>("chargeFractionMin")),
      centralMIPCharge_(iConfig.getParameter<double>("centralMIPCharge")),
      expSizeXAtLorentzAngleIncidence_(iConfig.getParameter<double>("expSizeXAtLorentzAngleIncidence")),
      expSizeXDeltaPerTanAlpha_(iConfig.getParameter<double>("expSizeXDeltaPerTanAlpha")),
      expSizeYAtNormalIncidence_(iConfig.getParameter<double>("expSizeYAtNormalIncidence")),
      tanLorentzAngle_(iConfig.getParameter<double>("tanLorentzAngle")),
      tanLorentzAngleBarrelLayer1_(iConfig.getParameter<double>("tanLorentzAngleBarrelLayer1")),
      chargeNorm_(iConfig.getParameter<double>("chargeNorm")),
      barrelTrain_(iConfig.getParameter<bool>("barrelTrain")),
      endcapTrain_(iConfig.getParameter<bool>("endcapTrain"))
{

  // Lock TFileService for thread safety
  usesResource("TFileService");
  edm::Service<TFileService> fileService;

   DeepCoreNtuplizerPhase2Tree= fileService->make<TTree>("DeepCoreNtuplizerPhase2Tree","DeepCoreNtuplizerPhase2Tree");
   DeepCoreNtuplizerPhase2Tree->Branch("cluster_measured",clusterMeas,Form("cluster_measured[%d][%d][%d]/D",jetDimX,jetDimY,Nlayer));
   DeepCoreNtuplizerPhase2Tree->Branch("trackPar", trackPar, Form("trackPar[%d][%d][%d][%d]/D",jetDimX,jetDimY,Nover,Npar+1)); //NOFLAG
   DeepCoreNtuplizerPhase2Tree->Branch("trackProb", trackProb, Form("trackProb[%d][%d][%d]/D",jetDimX,jetDimY,Nover));
   DeepCoreNtuplizerPhase2Tree->Branch("jet_eta",&jet_eta);
   DeepCoreNtuplizerPhase2Tree->Branch("jet_pt",&jet_pt);
   DeepCoreNtuplizerPhase2Tree->Branch("jet_p",&jet_p);
   DeepCoreNtuplizerPhase2Tree->Branch("jet_phi",&jet_phi);
   DeepCoreNtuplizerPhase2Tree->Branch("jet_Ntrack",&jet_Ntrack);
   DeepCoreNtuplizerPhase2Tree->Branch("track_pt",track_pt,"track_pt[100]/D");
   DeepCoreNtuplizerPhase2Tree->Branch("track_pz",track_pz,"track_pz[100]/D");
   DeepCoreNtuplizerPhase2Tree->Branch("track_eta",track_eta,"track_eta[100]/D");
   DeepCoreNtuplizerPhase2Tree->Branch("track_phi",track_phi,"track_phi[100]/D");
   DeepCoreNtuplizerPhase2Tree->Branch("event_ID",&eventID);

   //debug branches
   DeepCoreNtuplizerPhase2Tree->Branch("NPixLay",NPixLay, Form("NPixLay[%d]/I",Nlayer));





   /// reset variables to zero
   for(int ix=0; ix<jetDimX; ix++){
     for(int iy=0; iy<jetDimY; iy++){
       // first initialize clusterMeas
       for(unsigned int ilay=0; ilay < Nlayer; ilay++)
	 clusterMeas[ix][iy][ilay] = 0.0;
       for(unsigned int iOver=0; iOver < Nover; iOver++){
	 trackProb[ix][iy][iOver] =0.0;
	 for(unsigned int ipar=0; ipar<Npar+1; ipar++)
	   trackPar[ix][iy][iOver][ipar] =0.0;	 
       }
     }
   }

  
  if(barrelTrain_) {
    absEtaMax = 1.4;
  }
  if(endcapTrain_) {
    absEtaMin=1.4;
    // Phase-2 has 12 pixel disks per side (TFPX+TEPX, counted here as "layers" 5-16), but the image only
    // stores Nlayer=4 barrel layers, so endcap windows would be empty. A disk-to-image-layer mapping is
    // needed before endcap training makes sense.
    edm::LogWarning("DeepCoreNtuplizerPhase2") << "endcapTrain=True: the image stores only the 4 barrel layers; "
                                            "Phase-2 disks are not filled. Use barrelTrain for Phase-2.";
  }

  
}

DeepCoreNtuplizerPhase2::~DeepCoreNtuplizerPhase2() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
#define foreach BOOST_FOREACH

// ------------ method called to produce the data  ------------
void DeepCoreNtuplizerPhase2::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // for(jet){ //soglia pt 500 Gev
  //   save(jet->pt,eta)
  //   for(layer){
  //     find(inter = intersezione jet-layer)
  //     for(cluster on layer){
  //       if(cluser in finestra 30x30 intorno a inter){
  //         save(cluster->sig in posizione x,y,layer)
  //         (trackcluster[n],tracks[n])=split(cluster)
  //         for(trackcluster){ //cioè per ogni traccia che ha originato cluser
  //           save(trackcluster->sig in posizione x,y,layer,track)//con ampiezza tot cluster/Ntrack?
  //           save(trackcluster->track pt,eta,phi) //geom det, specific surface
  //         }
  //       }
  //     }
  //
  //   }
  // }

  magfield_   = iSetup.getHandle(theFieldToken);
  geometry_   = iSetup.getHandle(geomToken);
  propagator_ = iSetup.getHandle(propToken);


  iEvent.getByToken(pixelClusters_, inputPixelClusters);
  allSiPixelClusters.clear(); siPixelDetsWithClusters.clear();
  allSiPixelClusters.reserve(inputPixelClusters->dataSize()); // this is important, otherwise push_back invalidates the iterators

  edm::Handle<std::vector<SimTrack> > simtracks;
  iEvent.getByToken(simtracksToken, simtracks);

  Handle<std::vector<reco::Vertex> > vertices;
  iEvent.getByToken(vertices_, vertices);

  iEvent.getByToken(PSimHitToken, simhits);
  iEvent.getByToken(PSimHitECToken, simhitsEC);


  Handle<edm::View<reco::Candidate> > cores;
  iEvent.getByToken(cores_, cores);

  edm::ESHandle<PixelClusterParameterEstimator> parEstHandle = iSetup.getHandle(pixelClusterParameterEstimatorToken);
  const PixelClusterParameterEstimator* parEst;
  parEst = parEstHandle.product();

  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(tTopToken);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  auto output = std::make_unique<edmNew::DetSetVector<SiPixelCluster>>();



  edm::LogInfo("DeepCoreNtuplizerPhase2") << "number of jets in the event=" << cores->size();

  for (unsigned int ji = 0; ji < cores->size(); ji++) { //loop jet

    const reco::Candidate& jet = (*cores)[ji];
    
    if (jet.pt() < ptMin_)continue;
    if (jet.p()  < pMin_ )continue;
    if (std::abs(jet.eta()) <  absEtaMin)continue;
    if (std::abs(jet.eta()) >  absEtaMax)continue;
    edm::LogInfo("DeepCoreNtuplizerPhase2") << "jet in acceptance"
					 << ", pt= " << jet.pt()
					 << ", eta=" << jet.eta()
					 << ", p="   << jet.p(); 
      
 
    const reco::Vertex& jetVertex = (*vertices)[0];
    GlobalVector jetDirection(jet.px(), jet.py(), jet.pz());


    std::vector<GlobalVector> splitClustDirSet = splittedClusterDirections(jet, tTopo, parEst, jetVertex, 1);
    edm::LogInfo("DeepCoreNtuplizerPhase2") << "number of merged clusters on lay1="<<splitClustDirSet.size() << "+jetDir" ;
    if(splitClustDirSet.size()==0) {
        splitClustDirSet = splittedClusterDirections(jet, tTopo, parEst, jetVertex, 2);
        edm::LogInfo("DeepCoreNtuplizerPhase2") << "missing clusters on lay1, used lay2 to find directions, number of cluster=" << splitClustDirSet.size() << "+jetDir" << std::endl;
      }
      splitClustDirSet.push_back(jetDirection);

      for(int cc=0; cc<(int)splitClustDirSet.size(); cc++){
	for(uint lp=0; lp<Nlayer; lp++) NPixLay[lp] = 0;

	GlobalVector bigClustDir = splitClustDirSet.at(cc);


	LocalPoint jetInter(0,0,0);
	const auto & simtracksVector = simtracks.product();

	jet_pt = jet.pt();
	jet_p = jet.p();
	jet_eta = jet.eta();
	jet_phi = jet.phi();
	eventID= iEvent.id().event();

	auto jetVert = jetVertex; //trackInfo filling
	std::vector<std::map<int,SiPixelCluster>> clusterMapVector;
	std::vector<PSimHit> goodSimHit;

	edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt = inputPixelClusters->begin();

	// what to do if layer 1 is bad?
	const GeomDet* globDet = DetectorSelector(2, jet, bigClustDir, jetVertex, tTopo,parEst, simtracksVector); //select detector mostly hitten by the jet

	//fill the good det vector
	std::vector<const GeomDet*> goodDets;
	for(uint l=1; l<Nlayer+1; l++){
	  const GeomDet* goodDet = DetectorSelector(l, jet, bigClustDir, jetVertex, tTopo,parEst, simtracksVector);
	  goodDets.push_back(goodDet);	  
	}

	for (; detIt != inputPixelClusters->end(); detIt++) { //loop deset
	  const edmNew::DetSet<SiPixelCluster>& detset = *detIt;
	  const GeomDet* det = geometry_->idToDet(detset.id()); 

	  bool goodDetBool = false;
	  goodDetBool = (std::find(goodDets.begin(), goodDets.end(), det) != goodDets.end());
	  if (!pitchOK(det)) continue;  // window size in pixels assumes 25x100 um

	  for (auto cluster = detset.begin(); cluster != detset.end(); cluster++)
	    { //loop cluster
	      const SiPixelCluster& aCluster = *cluster;
	    //          det_id_type aClusterID= detset.id();
	    // if(DetId(aClusterID).subdetId()!=1) continue; #select only  barrel clusters
	      uint lay = tTopo->layer(det->geographicalId());
	      if(det->geographicalId().subdetId()==PixelSubdetector::PixelEndcap)
		lay=lay+4; //endcap layer counting = 5,6,7
	      
	      std::pair<bool, Basic3DVector<float>> interPair = findIntersection(bigClustDir,(reco::Candidate::Point)jetVertex.position(), det);

	      if(interPair.first==false) continue;

	      Basic3DVector<float> inter = interPair.second;

	      auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);
	      LocalPoint cPos_local = parEst->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt->id())))[0].first;

	      edm::LogInfo("DeepCoreNtuplizerPhase2").log([&](auto & logger){
		GlobalPoint pointVertex(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
		GlobalPoint cPos = det->surface().toGlobal(parEst->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt->id())))[0].first);
		// GlobalVector intersectionDir = (GlobalPoint) inter - pointVertex;
              // auto deltaR_glob = Geom::deltaR(bigClustDir, intersectionDir);
		GlobalVector clusterDir = cPos - pointVertex;
		auto deltaR_clust = Geom::deltaR(bigClustDir, clusterDir);
		logger << "cluster with DeltaR(clust,window center direction)" << deltaR_clust;
	      });

	    // look for 30 x 30 pixels around the cluster direction
	      if(std::abs(cPos_local.x()-localInter.x())/pitchX > jetDimX/2)
		continue;
	      if(std::abs(cPos_local.y()-localInter.y())/pitchY > jetDimY/2)
		continue;
	      
	      if (lay >= 1 && lay <= Nlayer)  // endcap "layers" are 5-16 in Phase-2: outside NPixLay[Nlayer]
	        NPixLay[lay-1]= NPixLay[lay-1]+1;
	      if(goodDetBool)
		fillPixelMatrix(aCluster,lay,localInter, det);
	    } //cluster
	} //detset 


      // -----good sim hit selection (hit inside window) -----------//
	std::vector<PSimHit>::const_iterator shIt = simhits->begin();

      // std::set<const GeomDet*> simhitsDetSet;
      for (; shIt != simhits->end(); shIt++) { //loop deset
        // const edmNew::DetSet<PSimHit>& detset = *shIt;
        const GeomDet* det = geometry_->idToDet((*shIt).detUnitId());
        if(det!=globDet) continue;
        std::pair<bool, Basic3DVector<float>> interPair = findIntersection(bigClustDir,(reco::Candidate::Point)jetVertex.position(), det);
        if(interPair.first==false) continue;
        Basic3DVector<float> inter = interPair.second;
        auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);
        if(jetInter.x()==0 && jetInter.y()==0 && jetInter.z()==0) jetInter = localInter; //filling infoTracks
	// look for simHits within 30x30 pixels
        if(std::abs(((*shIt).localPosition()).x()-localInter.x())/pitchX>jetDimX/2)
	  continue;
	if(std::abs(((*shIt).localPosition()).y()-localInter.y())/pitchY>jetDimY/2)
	  continue;
	edm::LogInfo("DeepCoreNtuplizerPhase2") << "good sim hit on layer 2,  track ID" << (*shIt).trackId();
	goodSimHit.push_back((*shIt));
      
      } // end of loop over simHits

      fillTrackInfo(jet, simtracksVector, jetInter, bigClustDir, jetVert, globDet, parEst, goodSimHit,tTopo);

      edm::LogInfo("DeepCoreNtuplizerPhase2") << "Number of good simHits on Layer 2 = " << goodSimHit.size();

      clusterMapVector.clear();
      DeepCoreNtuplizerPhase2Tree->Fill();
      edm::LogInfo("DeepCoreNtuplizerPhase2").log([&](auto& li) {
	li << "N.pixel on layer ";
	for (size_t i = 0; i < Nlayer; ++i)
	  li << i << "=" << NPixLay[i] << ", ";
	li << "\n";
      });    

      // reset to zero 
      for(int ix=0; ix<jetDimX; ix++){
	for(int iy=0; iy<jetDimY; iy++){	
	  for(unsigned int ilay=0; ilay < Nlayer; ilay++)
	    clusterMeas[ix][iy][ilay] = 0.0;	
	  for(unsigned int iOver=0; iOver<Nover; iOver++){
	    trackProb[ix][iy][iOver] =0.0;
	    for(unsigned int ipar=0; ipar < Npar+1; ipar++)
	      trackPar[ix][iy][iOver][ipar] =0.0; //NOFLAG
	  } 
	}
      }
      for(unsigned int h=0; h<Ntrack; h++){
	track_pt[h] = 0.0;
	track_pz[h] = 0.0;
	track_eta[h] = 0.0;
	track_phi[h] = 0.0;
      }    
      } // end of loop over big cluster directions
  } // end of loop over jets

}




std::pair<bool, Basic3DVector<float>> DeepCoreNtuplizerPhase2::findIntersection(const GlobalVector & dir,const  reco::Candidate::Point & vertex, const GeomDet* det){
  StraightLinePlaneCrossing vertexPlane(Basic3DVector<float>(vertex.x(),vertex.y(),vertex.z()), Basic3DVector<float>(dir.x(),dir.y(),dir.z()));

  if (!det) return {false, Basic3DVector<float>(0, 0, 0)};

  // only crossings in front of the vertex: the full line also crosses the module plane on the opposite
  // side of the detector (phi+pi, -eta), which could otherwise be selected by DetectorSelector
  const auto s = vertexPlane.pathLength(det->specificSurface());
  if (!s.first || s.second <= 0) return {false, Basic3DVector<float>(0, 0, 0)};
  return {true, vertexPlane.position(s.second)};
}

bool DeepCoreNtuplizerPhase2::pitchOK(const GeomDet* det){
  const auto* pdu = dynamic_cast<const PixelGeomDetUnit*>(det);
  if (!pdu) return false;
  const auto p = pdu->specificTopology().pitch();
  const bool ok = std::abs(p.first - pitchX) < 1e-5 && std::abs(p.second - pitchY) < 1e-5;
  if (!ok && !warnedPitch_) {
    edm::LogWarning("DeepCoreNtuplizerPhase2") << "module " << det->geographicalId().rawId() << " has pitch "
                                         << p.first * 1e4 << " x " << p.second * 1e4
                                         << " um, not 25 x 100 um; such modules are skipped (window size in pixels)";
    warnedPitch_ = true;
  }
  return ok;
}



std::pair<int,int> DeepCoreNtuplizerPhase2::local2Pixel(double locX, double locY, const GeomDet* det){
  LocalPoint locXY(locX,locY);
  float pixX=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).first;
  float pixY=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).second;
  std::pair<int, int> out(pixX,pixY);
  return out;
}

LocalPoint DeepCoreNtuplizerPhase2::pixel2Local(int pixX, int pixY, const GeomDet* det){
  float locX=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localX(pixX);
  float locY=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localY(pixY);
  LocalPoint locXY(locX,locY);
  return locXY;
}
 
// pixel size fix
std::pair<bool,bool> DeepCoreNtuplizerPhase2::pixel2Size(int pixX, int pixY, const GeomDet* det){
  bool pix_x_size=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().isItBigPixelInX(pixX);
  bool pix_y_size=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().isItBigPixelInY(pixY);
  std::pair<bool,bool> pix_xy_size(pix_x_size,pix_y_size);
  return pix_xy_size;
}

int DeepCoreNtuplizerPhase2::pixelFlipper(const GeomDet* det){
  int out =1;
  LocalVector locZdir(0,0,1);
  GlobalVector globZdir  = det->specificSurface().toGlobal(locZdir);
  GlobalPoint globDetCenter = det->position();
  float direction = globZdir.x()*globDetCenter.x()+ globZdir.y()*globDetCenter.y()+ globZdir.z()*globDetCenter.z();
  //float direction = globZdir.dot(globDetCenter);
  if(direction<0) out =-1;
  // out=1;
  return out;
}



void DeepCoreNtuplizerPhase2::fillPixelMatrix(const SiPixelCluster & cluster, uint layer, auto inter, const GeomDet* det){

  int flip = pixelFlipper(det); // 1=not flip, -1=flip

  for(int i=0; i<cluster.size();i++){
    SiPixelCluster::Pixel pix = cluster.pixel(i);
    std::pair<int,int> pixInter = local2Pixel(inter.x(),inter.y(),det);
    int nx = pix.x-pixInter.first;
    int ny = pix.y-pixInter.second;
    nx=flip*nx;

    if(abs(nx)<jetDimX/2 && abs(ny)<jetDimY/2){
      nx = nx+jetDimX/2;
      ny = ny+jetDimY/2;
      if(nx<jetDimX && ny<jetDimY && layer>=1 && layer-1< Nlayer && nx>=0 && ny>=0) clusterMeas[nx][ny][layer-1] += (pix.adc)/(float)(chargeNorm_);
    }
  }

}

void DeepCoreNtuplizerPhase2::fillTrackInfo(const reco::Candidate& jet, const auto & stVector, auto jti, auto jetDir, auto jVert, const GeomDet* det, const PixelClusterParameterEstimator* cpe,  std::vector<PSimHit> goodSimHit, const TrackerTopology* const tTopo){

  bool oneHitInfo = false; //track par given only where prob=1

  struct trkInfoObj
  {
    int prob;
    double dist;
    double xpos;
    double ypos;
    double xangle;
    double yangle;
    int zero_flag; //useful for CNN training only: 0 if x,y,eta,phi==0
    double one_over_pt;
    trkInfoObj(int pp, double dd, double xx, double yy, double tx, double ty, int zf, double ptInv) : //, double jeta, double jpt) : // double xd, double yd, double n) :
      prob(pp),
      dist(dd),
      xpos(xx),
      ypos(yy),
      xangle(tx),
      yangle(ty),
      zero_flag(zf),
      one_over_pt(ptInv) {}

    bool operator < (const trkInfoObj& trk) const
    {
      return (dist < trk.dist);
    }
  };

  std::vector<SimTrack> goodSimTrk;

  for(uint j=0; j<stVector->size(); j++){ //matched tracks selection and vertex assigment

    SimTrack st = stVector->at(j);
    if(st.momentum().Pt()<1 or st.momentum().Pt()>100000) continue; //higher cut is to avoid problematic tracks
    bool matchGoodSimHit=false;
    for(std::vector<PSimHit>::const_iterator it=goodSimHit.begin(); it!=goodSimHit.end(); ++it) {
      // same Geant4 track AND same (pile-up) event AND same particle (hits of un-saved secondaries carry the
      // parent's trackId)
      if(st.trackId()==(*it).trackId() && st.eventId()==(*it).eventId() && st.type()==(*it).particleType()) {
	matchGoodSimHit=true;
	break;
      }
    }
    if(matchGoodSimHit)
      goodSimTrk.push_back(st);
  }  

  
  jet_Ntrack=goodSimTrk.size();
  for(uint j=0; j<goodSimTrk.size() && j<Ntrack; j++){
    SimTrack st = goodSimTrk.at(j);
    track_pt[j] = st.momentum().Pt();
    track_pz[j] = st.momentum().Pz();
    track_eta[j] = st.momentum().Eta();
    track_phi[j] = st.momentum().Phi();
  }


  for(int x=0; x<jetDimX; x++){
    for(int y=0; y<jetDimY; y++){
      std::vector<trkInfoObj> tracksInfo;
      for(uint j=0; j<goodSimTrk.size(); j++){
	SimTrack st = goodSimTrk.at(j);
	GlobalVector trkMom(st.trackerSurfaceMomentum().x(),st.trackerSurfaceMomentum().y(), st.trackerSurfaceMomentum().z());
	int flip = pixelFlipper(det); // 1=not flip, -1=flip

	PSimHit theSimHit;
	for(uint sh=0; sh<goodSimHit.size(); sh++){
	  if(goodSimHit[sh].trackId()==st.trackId() && goodSimHit[sh].eventId()==st.eventId() &&
	     goodSimHit[sh].detUnitId()==det->geographicalId().rawId()){
	    theSimHit = goodSimHit[sh];
	  }
	}

	LocalPoint localTrkInter(flip*(theSimHit.localPosition()).x(),(theSimHit.localPosition()).y(),(theSimHit.localPosition()).z());
	LocalPoint Jinter(flip*jti.x(),jti.y(),jti.z());

	std::pair<int,int> pixJInter = local2Pixel(Jinter.x(),Jinter.y(),det);
	std::pair<int,int> pixTrkInter = local2Pixel(localTrkInter.x(),localTrkInter.y(),det); //ADDED PITCH

	int pixX = (pixTrkInter.first-pixJInter.first);
	int pixY = (pixTrkInter.second-pixJInter.second);

	pixX = pixX+jetDimX/2;
	pixY = pixY+jetDimY/2;

	double info[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	if (x==pixX && y==pixY)
	  info[0]= 1;         
	else info[0] = 0;
          
        // pixel centre from the module topology (handles the larger pixels at chip edges of Phase-2 modules,
        // instead of assuming big pixel = 2 x pitch). localX(i+0.5) is the centre of pixel i.
          const auto& ptopo = dynamic_cast<const PixelGeomDetUnit*>(det)->specificTopology();
          const int ipx = x + pixJInter.first - jetDimX/2;
          const int ipy = y + pixJInter.second - jetDimY/2;
          LocalPoint pixCenter(ptopo.localX(ipx + 0.5f), ptopo.localY(ipy + 0.5f));
          double distX =  localTrkInter.x()-pixCenter.x();
          double distY =  localTrkInter.y()-pixCenter.y();
         
          // debug
         // std::cout << " dx [cm]= " << distX << " Track crossing ( " << localTrkInter.x() << ") - pixel location ( " << pix2loc.x() << " ) - 0.005 " << std::endl;
         // std::cout << " is x big? " << sz_xy.first << " is y big? " << sz_xy.second << std::endl;
         // std::cout << " dy [cm]= " << distY << " Track crossing ( " << localTrkInter.y() << ") - pixel location ( " << pix2loc.y() << " ) - 0.0075 " << std::endl;

	  if(distX<0.000001 && distX> -0.000001)
	    distX = 0.0;
	  
	  if(distY<0.000001 && distY> - 0.000001)
	    distY = 0.0;
	  
          info[1] = sqrt(distX*distX+distY*distY);

          if(fabs(distX)<distThr*pitchX && fabs(distY)<distThr*pitchY){

            info[2] = distX;
            info[3] = distY;

            info[4] = st.momentum().Eta()-jetDir.eta();
            info[5] = deltaPhi(st.momentum().Phi(),jetDir.phi());
            info[7] = st.momentum().Pt(); // 1/pt --> pt fix
          }
          else{
          if(info[0]== 1) {
             edm::LogError("DeepCoreNtuplizerPhase2") << "prob=1 outside of radious of " << distThr << "pixel from track intersection, probable bug";
          }
          info[2] = 99999999;
          info[3] = 99999999;
          }
          
          if(info[0]==0 && info[2]==0.0 && info[3]==0.0 && info[4]==0.0 && info[5]==0.0) {
            info[6] = 0;
          }
          else if(info[2] > 9999999 && info[3] > 9999999) {
            info[6] = 0;
          }
          else{
             info[6]=1;
           }

          tracksInfo.push_back(trkInfoObj(info[0],info[1],info[2],info[3],info[4],info[5],info[6],info[7]));

      }//good sim tracks

      std::sort(tracksInfo.begin(), tracksInfo.end());

    
      for(uint trk=0; trk< Nover && trk<tracksInfo.size(); trk++){

	if(tracksInfo.at(trk).xpos!=99999999)
	  trackPar[x][y][trk][0]=100*tracksInfo.at(trk).xpos;
	else trackPar[x][y][trk][0]=0.0;
	
	if(tracksInfo.at(trk).ypos!=99999999)
	  trackPar[x][y][trk][1]=100*tracksInfo.at(trk).ypos;
	else trackPar[x][y][trk][1]=0.0;

	trackPar[x][y][trk][2]=100*tracksInfo.at(trk).xangle;
	trackPar[x][y][trk][3]=100*tracksInfo.at(trk).yangle;
	trackPar[x][y][trk][4]=tracksInfo.at(trk).one_over_pt;
	trackPar[x][y][trk][5]=tracksInfo.at(trk).zero_flag; //NOFLAG
	trackProb[x][y][trk]=tracksInfo.at(trk).prob;

	if(oneHitInfo){
	  if(trackProb[x][y][trk]==0) {
	    for(uint pp=0; pp<Npar+1; pp++)
	      trackPar[x][y][trk][pp] = 0.0;
	  }
	}

	if(tracksInfo.at(trk).zero_flag!=0)
	  edm::LogInfo("DeepCoreNtuplizerPhase2") << "Filling track info with zero-flag!=0, track=" << trk << ", prob="<<tracksInfo.at(trk).prob<< ", x bin="<< x << ", y bin=" << y<< ", "<<  "x position=" << tracksInfo.at(trk).xpos << " y position=" << tracksInfo.at(trk).ypos << std::endl;
      }//trk (filling)
      tracksInfo.clear();
    } // y
  }//x
  goodSimTrk.clear();
}


const GeomDet* DeepCoreNtuplizerPhase2::DetectorSelector(int llay, const reco::Candidate& jet, GlobalVector jetDir, const reco::Vertex& jetVertex, const TrackerTopology* const tTopo, const PixelClusterParameterEstimator* parEst, const auto & simtracksVector){
  struct distCompare {
    bool operator()(std::pair<double,const GeomDet*> x, std::pair<double,const GeomDet*> y) const
    {return x.first < y.first;} //order in ascending distance
  };

  std::set<std::pair<double,const GeomDet*>, distCompare> distDetSet;//distance square from center of det
  LocalPoint jetInter(0,0,0);

  std::vector<PSimHit>::const_iterator shIt = simhits->begin();
  std::vector<PSimHit>::const_iterator shItEC = simhitsEC->begin();

  std::set<const GeomDet*> simhitsDetSet;

  for (; shIt != simhits->end(); shIt++) { //loop on simit to find correspondent det (barrel)
    const GeomDet* det = geometry_->idToDet((*shIt).detUnitId());
    simhitsDetSet.insert(det);
  }

  for (; shItEC != simhitsEC->end(); shItEC++) { //loop on simit to find correspondent det (endCaps)
    const GeomDet* det = geometry_->idToDet((*shItEC).detUnitId());
    simhitsDetSet.insert(det);
  }

  for (std::set<const GeomDet*>::iterator detIt=simhitsDetSet.begin(); detIt != simhitsDetSet.end(); detIt++) { //loop deset

    const GeomDet* det = *detIt;
    
    int lay = tTopo->layer(det->geographicalId());
    if(det->geographicalId().subdetId()==PixelSubdetector::PixelEndcap)
      lay=lay+4; //endcap layer counting = 5,6,7
        
    std::pair<bool, Basic3DVector<float>> interPair = findIntersection(jetDir,(reco::Candidate::Point)jetVertex.position(), det);
    if(interPair.first==false) continue;
    Basic3DVector<float> inter = interPair.second;
    auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);

    if(lay==llay) {
      double dist2 = localInter.x()*localInter.x()+localInter.y()*localInter.y();
      std::pair<double,const GeomDet*> distDet(dist2, det);
      distDetSet.insert(distDet);
    }

  } //detset

  if(distDetSet.size()!=0) edm::LogInfo("DeepCoreNtuplizerPhase2") << "Intersecting detector, layer ="<< llay << "  det=" << distDetSet.begin()->second->gdetIndex() << ", distance =" << distDetSet.begin()->first << std::endl;

  if(distDetSet.size()==0) return (GeomDet*)0;
  else 
      return distDetSet.begin()->second;
    
}

std::vector<GlobalVector> DeepCoreNtuplizerPhase2::splittedClusterDirections(const reco::Candidate& jet, const TrackerTopology* const tTopo, auto parEst, const reco::Vertex& jetVertex , int layer){
  std::vector<GlobalVector> clustDirs;
  edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt_int = inputPixelClusters->begin();

  for (; detIt_int != inputPixelClusters->end(); detIt_int++) {

    const edmNew::DetSet<SiPixelCluster>& detset_int = *detIt_int;
    const GeomDet* det_int = geometry_->idToDet(detset_int.id());
    int lay = tTopo->layer(det_int->geographicalId());
    bool isEndCap =(det_int->geographicalId().subdetId()==PixelSubdetector::PixelEndcap);
      
    if(isEndCap)
      lay=lay+4; //endcap layer counting = 5,6,7
    if(lay != layer) continue; //NB: saved bigclusetr on all the layers
    
    for (auto cluster = detset_int.begin(); cluster != detset_int.end(); cluster++) {
      const SiPixelCluster& aCluster = *cluster;
      GlobalPoint cPos = det_int->surface().toGlobal(parEst->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt_int->id())))[0].first);
      GlobalPoint ppv(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
      GlobalVector clusterDir = cPos - ppv;
      GlobalVector jetDir(jet.px(), jet.py(), jet.pz());
      
      if (Geom::deltaR(jetDir, clusterDir) > deltaR_)continue;
      // check if the cluster has to be splitted: same test as RecoLocalTracker/SubCollectionProducers
      // JetCoreClusterSplitter, with the jet direction in the module frame and the sensor thickness and pitch
      // taken from the geometry (the Phase-1 constants 1.3/1.9 assumed 285 um sensors and 150 um pitch in y)
      const LocalVector jetDirLocal = det_int->surface().toLocal(jetDir);
      const float jetTanAlpha = jetDirLocal.x() / jetDirLocal.z();
      const float jetTanBeta = jetDirLocal.y() / jetDirLocal.z();
      const float jetZOverRho = std::sqrt(jetTanAlpha * jetTanAlpha + jetTanBeta * jetTanBeta);
      const auto pitch = static_cast<const PixelTopology&>(det_int->topology()).pitch();
      const float thickness = det_int->surface().bounds().thickness();
      float tanLA = tanLorentzAngle_;
      if (!isEndCap && tTopo->pxbLayer(det_int->geographicalId()) == 1)
	tanLA = tanLorentzAngleBarrelLayer1_;
      float expSizeX = expSizeXAtLorentzAngleIncidence_ + std::abs(expSizeXDeltaPerTanAlpha_ * (jetTanAlpha - tanLA));
      float expSizeY = std::sqrt(expSizeYAtNormalIncidence_ * expSizeYAtNormalIncidence_ +
				 thickness * thickness / (pitch.second * pitch.second) * jetTanBeta * jetTanBeta);
      if (expSizeX < 1.f) expSizeX = 1.f;
      if (expSizeY < 1.f) expSizeY = 1.f;
      float expCharge =
	std::sqrt(1.08f + jetZOverRho * jetZOverRho) * centralMIPCharge_;
      if (aCluster.charge() > expCharge * chargeFracMin_ && (aCluster.sizeX() > expSizeX + 1 ||  aCluster.sizeY() > expSizeY + 1)) {
	edm::LogInfo("DeepCoreNtuplizerPhase2") << "found mergedcluster with deltaR" << Geom::deltaR(jetDir, clusterDir)<< ", on layer=" <<lay << std::endl;
	clustDirs.push_back(clusterDir);
      }
    } // end of loop over clusters
  } // end of loop over detId
  return clustDirs;
}



// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DeepCoreNtuplizerPhase2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("pixelClusters", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<edm::InputTag>("cores", edm::InputTag("ak4CaloJets"));
  desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("simHit", edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"));
  desc.add<edm::InputTag>("simHitEC", edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"));
  desc.add<double>("ptMin", 500);
  desc.add<double>("pMin", 0);
  desc.add<double>("deltaR", 0.1);
  // Phase-2 values of the jet-core cluster splitter (RecoTracker/IterativeTracking InitialStepPreSplitting_cff,
  // splitClustersInPhase2Pixel): 150 um sensors give about half the Phase-1 MIP charge
  desc.add<double>("centralMIPCharge", 12000.0);
  desc.add<double>("chargeFractionMin", 2);
  desc.add<double>("expSizeXAtLorentzAngleIncidence", 1.5);
  desc.add<double>("expSizeXDeltaPerTanAlpha", 5.2);
  desc.add<double>("expSizeYAtNormalIncidence", 2.3);
  desc.add<double>("tanLorentzAngle", -0.21);
  desc.add<double>("tanLorentzAngleBarrelLayer1", 0.0);
  desc.add<double>("chargeNorm", 14000.)
      ->setComment("pixel charge normalization in the image; the seed generator must use the same value");
  desc.add<std::string>("pixelCPE", "PixelCPEGeneric");
  desc.add<bool>("barrelTrain", true);
  desc.add<bool>("endcapTrain", false);
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepCoreNtuplizerPhase2);
