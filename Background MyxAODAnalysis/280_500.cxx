// Infrastructure include(s):
#include "xAODRootAccess/Init.h"
#include "xAODRootAccess/TEvent.h"
#include <EventLoop/Job.h>
#include <EventLoop/StatusCode.h>
#include <EventLoop/Worker.h>
#include <MyAnalysis/MyxAODAnalysis.h>
#include "xAODJet/JetContainer.h"
#include "JetSubStructureUtils/BosonTag.h"  // if we are using the boson tag
// ASG status code check
#include <AsgTools/MessageCheck.h>
// EDM includes:
#include "xAODEventInfo/EventInfo.h"

#include <TH1.h>
#include <TH2.h>
#include "xAODMuon/MuonContainer.h"
#include "xAODEgamma/PhotonContainer.h"
#include "xAODEgamma/Photon.h"
#include "xAODTruth/TruthParticle.h"
#include "xAODTruth/TruthParticleContainer.h"
#include <TLorentzVector.h>
#include <math.h>
#include <iostream>
#include <vector>

// this is needed to distribute the algorithm to the workers
ClassImp(MyxAODAnalysis)



MyxAODAnalysis :: MyxAODAnalysis ()
{
    // Here you put any code for the base initialization of variables,
    // e.g. initialize all pointers to 0.  Note that you should only put
    // the most basic initialization here, since this method will be
    // called on both the submission and the worker node.  Most of your
    // initialization code will go into histInitialize() and
    // initialize().
}



EL::StatusCode MyxAODAnalysis :: setupJob (EL::Job& job)
{
    // Here you put code that sets up the job on the submission object
    // so that it is ready to work with your algorithm, e.g. you can
    // request the D3PDReader service or add output files.  Any code you
    // put here could instead also go into the submission script.  The
    // sole advantage of putting it here is that it gets automatically
    // activated/deactivated when you add/remove the algorithm from your
    // job, which may or may not be of value to you.
    // let's initialize the algorithm to use the xAODRootAccess package
	job.useXAOD ();
    ANA_CHECK_SET_TYPE (EL::StatusCode); // set type of return code you are expecting (add to top of each function once)
    ANA_CHECK(xAOD::Init());
    return EL::StatusCode::SUCCESS;
}



EL::StatusCode MyxAODAnalysis :: histInitialize ()
{
    // Here you do everything that needs to be done at the very
    // beginning on each worker node, e.g. create histograms and output
    // trees.  This method gets called before any input files are
    // connected.
	h0 = new TH1F("h0", "#gamma Pt",50,280*0.8,500*1.2);
	h1 = new TH1F("h1", "#eta_{#gamma}",100,-4,4);
	h2 = new TH2C("h2","#gamma Pt and #eta Correlation",100,-4,4,50,280*0.8,500*1.2);
	h3 = new TH1F("h3", "#varphi_{#gamma}",100,-4,4);

	h4 = new TH1F("h4", "Pt of Highest Pt Jet",100,-150,1500);
	h5 = new TH1F("h5", "#eta of Highest Pt Jet",100,-5.5,5.5);
	h6 = new TH1F("h6", "#varphi of Highest Pt Jet",100,-5.5,5.5);
	h7 = new TH1F("h7", "Jet Mass of Highest Pt Jet", 100, -11*TMath::Power(10,3), 400000);
    h8 = new TH1F("h8", "ECF2 of Highest Pt Jet",100,-6*TMath::Power(10,9),80*TMath::Power(10,9));
    h9 = new TH1F("h9", "ECF3 of Highest Pt Jet",100,-60*TMath::Power(10,12),300*TMath::Power(10,12));
    h10 = new TH1F("h10", "D2 of Highest Pt Jet",100,-1,5);
    h11 = new TH1I("h11", "Number of Jets in Highest Pt Jet",21,-6,15);

	h12 = new TH1F("h12", "Pt of Second Highest Pt Jet",100,-150,1500);
	h13 = new TH1F("h13", "#eta of Second Highest Pt Jet",100,-5.5,5.5);
	h14 = new TH1F("h14", "#varphi of Second Highest Pt Jet",100,-5.5,5.5);
	h15 = new TH1F("h15", "Jet Mass of Second Highest Pt Jet", 100, -11*TMath::Power(10,3), 200*TMath::Power(10,3));
    h16 = new TH1F("h16", "ECF2 of Second Highest Pt Jet",100,-6*TMath::Power(10,9),20*TMath::Power(10,9));
    h17 = new TH1F("h17", "ECF3 of Second Highest Pt Jet",100,-60*TMath::Power(10,12),150*TMath::Power(10,12));
    h18 = new TH1F("h18", "D2 of Second Highest Pt Jet",100,-1,5);
    h19 = new TH1I("h19", "Number of Jets in Second Highest Pt Jet",21,-6,15);

    h20 = new TH1F("h20", "Pt of Third Highest Pt Jet",100,-150,400);
    h21 = new TH1F("h21", "#eta of Third Highest Pt Jet",100,-5.5,5.5);
    h22 = new TH1F("h22", "#varphi of Third Highest Pt Jet",100,-5.5,5.5);
    h23 = new TH1F("h23", "Jet Mass of Third Highest Pt Jet", 100, -11*TMath::Power(10,3), 100*TMath::Power(10,3));
    h24 = new TH1F("h24", "ECF2 of Third Highest Pt Jet",100,-6*TMath::Power(10,9),10*TMath::Power(10,9));
    h25 = new TH1F("h25", "ECF3 of Third Highest Pt Jet",100,-60*TMath::Power(10,12),100*TMath::Power(10,12));
    h26 = new TH1F("h26", "D2 of Third Highest Pt Jet",100,-1,5);
    h27 = new TH1I("h27", "Number of Jets in Third Highest Pt Jet",21,-6,15);

	wk()->addOutput (h0);
    wk()->addOutput (h1);
    wk()->addOutput (h2);
    wk()->addOutput (h3);
    wk()->addOutput (h4);
    wk()->addOutput (h5);
    wk()->addOutput (h6);
    wk()->addOutput (h7);
    wk()->addOutput (h8);
    wk()->addOutput (h9);
    wk()->addOutput (h10);
    wk()->addOutput (h11);
    wk()->addOutput (h12);
    wk()->addOutput (h13);
    wk()->addOutput (h14);
    wk()->addOutput (h15);
    wk()->addOutput (h16);
    wk()->addOutput (h17);
    wk()->addOutput (h18);
    wk()->addOutput (h19);
    wk()->addOutput (h20);
    wk()->addOutput (h21);
    wk()->addOutput (h22);
    wk()->addOutput (h23);
    wk()->addOutput (h24);
    wk()->addOutput (h25);
    wk()->addOutput (h26);
    wk()->addOutput (h27);

	return EL::StatusCode::SUCCESS;
}



EL::StatusCode MyxAODAnalysis :: fileExecute ()
{
    // Here you do everything that needs to be done exactly once for every
    // single file, e.g. collect a list of all lumi-blocks processed
	return EL::StatusCode::SUCCESS;
}



EL::StatusCode MyxAODAnalysis :: changeInput (bool firstFile)
{
    // Here you do everything you need to do when we change input files,
    // e.g. resetting branch addresses on trees.  If you are using
    // D3PDReader or a similar service this method is not needed.
	return EL::StatusCode::SUCCESS;
}


EL::StatusCode MyxAODAnalysis :: initialize ()
{
    // Here you do everything that you need to do after the first input
    // file has been connected and before the first event is processed,
    // e.g. create additional histograms based on which variables are
    // available in the input files.  You can also create all of your
    // histograms and trees in here, but be aware that this method
    // doesn't get called if no events are processed.  So any objects
    // you create here won't be available in the output if you have no
    // input events.
    ANA_CHECK_SET_TYPE (EL::StatusCode); // set type of return code you are expecting (add to top of each function once)
    xAOD::TEvent* event = wk()->xaodEvent();
    
    total_eventCounter = 0;
    background = fopen("./280_500background.txt", "a");
    // signal = fopen("./280_500signal.txt", "a");
    return EL::StatusCode::SUCCESS;
}



EL::StatusCode MyxAODAnalysis :: execute ()
{

    ANA_CHECK_SET_TYPE (EL::StatusCode); // set type of return code you are expecting (add to top of each function once)
    xAOD::TEvent* event = wk()->xaodEvent();
    const xAOD::EventInfo* eventInfo = 0;
    ANA_CHECK(event->retrieve(eventInfo, "EventInfo"));


    // Find highest Pt photon
    float photon_highest_pt = 0;
    const xAOD::PhotonContainer* photons = 0;
    TLorentzVector photon, pt_photon;
    ANA_CHECK( event->retrieve(photons, "Photons") );
    xAOD::PhotonContainer::const_iterator photon_itr = photons->begin();
    xAOD::PhotonContainer::const_iterator photon_itr_end = photons->end();
    for( ; photon_itr != photon_itr_end; photon_itr ++ ){
        photon = (*photon_itr)->p4();
        if(photon.Pt()>photon_highest_pt){
            photon_highest_pt = photon.Pt();
            pt_photon = photon;
        }
    }

    // Find 3 highest Pt jet
    // Some jets will be empty if there are not even 3 jets in total

    int jet_counter = 0;

    total_eventCounter++;

    float aux_jet_pt_1 = 0;
    float aux_jet_pt_2 = 0;
    float aux_jet_pt_3 = 0;
    TLorentzVector aux_jet_p4_1, aux_jet_p4_2, aux_jet_p4_3;

    const xAOD::JetContainer* aux_jets = 0;
    ANA_CHECK( event->retrieve(aux_jets, "AntiKt10LCTopoTrimmedPtFrac5SmallR20Jets") );
    xAOD::JetContainer::const_iterator aux_jets_itr = aux_jets->begin();
    xAOD::JetContainer::const_iterator aux_jets_itr_end = aux_jets->end();
    xAOD::JetContainer::const_iterator aux_jets_itr_1, aux_jets_itr_2, aux_jets_itr_3;
    for( ; aux_jets_itr != aux_jets_itr_end; aux_jets_itr ++ ){

        float distance_from_photon = TMath::Sqrt(TMath::Power((*aux_jets_itr)->p4().Eta()-pt_photon.Eta(),2)+TMath::Power((*aux_jets_itr)->p4().Phi()-pt_photon.Phi(),2));

        if (distance_from_photon>1){
            // printf("Event %d, jets are:\n (%f, %f, %f, %f, %f, %f)\n",total_eventCounter,
            //     (*aux_jets_itr)->p4().Pt()/1000,(*aux_jets_itr)->p4().Eta(),(*aux_jets_itr)->p4().Phi(),(*aux_jets_itr)->p4().M(),(*aux_jets_itr)->auxdata<float>("ECF2"),(*aux_jets_itr)->auxdata<float>("ECF3"));
        }

        if(distance_from_photon>1){
            jet_counter++;
            if((*aux_jets_itr)->p4().Pt()>aux_jet_pt_1){
                aux_jet_pt_3 = aux_jet_pt_2;
                aux_jet_pt_2 = aux_jet_pt_1;
                aux_jet_pt_1 = (*aux_jets_itr)->p4().Pt();

                aux_jets_itr_3 = aux_jets_itr_2;
                aux_jets_itr_2 = aux_jets_itr_1;
                aux_jets_itr_1 = aux_jets_itr;
            }
            else if((*aux_jets_itr)->p4().Pt()>aux_jet_pt_2){
                aux_jet_pt_3 = aux_jet_pt_2;
                aux_jet_pt_2 = (*aux_jets_itr)->p4().Pt();

                aux_jets_itr_3 = aux_jets_itr_2;
                aux_jets_itr_2 = aux_jets_itr;
            }
            else if((*aux_jets_itr)->p4().Pt()>aux_jet_pt_3){
                aux_jet_pt_3 = (*aux_jets_itr)->p4().Pt();

                aux_jets_itr_3 = aux_jets_itr;
            }
        }
    }

    if(jet_counter<3 && total_eventCounter<=500){
        printf("Event %d has %d jets:\n",total_eventCounter, jet_counter);
        // exit(0);
    }

    // D2 = ECF3*pow(ECF1,3.0)/pow(ECF2,3.0)

    // Jet 1
    float D2_1 = -1;
    float subjets_1 = -5;
    float ECF2_1 = -5*TMath::Power(10,9);
    float ECF3_1 = -50*TMath::Power(10,12);
    float pt_1 = -100;
    float eta_1 = -5;
    float phi_1 = -5;
    float m_1 = -10*TMath::Power(10,3);

    if(jet_counter>=1){
    	if((*aux_jets_itr_1)->auxdata<float>("ECF2") != 0){
    		D2_1 = (*aux_jets_itr_1)->auxdata<float>("ECF3");
    		D2_1 = D2_1*pow((*aux_jets_itr_1)->auxdata<float>("ECF1"),3);
    		D2_1 = D2_1/pow((*aux_jets_itr_1)->auxdata<float>("ECF2"),3);
    	}
    	subjets_1 = (float) (*aux_jets_itr_1)->auxdata<int>("NTrimSubjets");
    	ECF2_1 = (*aux_jets_itr_1)->auxdata<float>("ECF2");
    	ECF3_1 = (*aux_jets_itr_1)->auxdata<float>("ECF3");
    	pt_1 = (*aux_jets_itr_1)->p4().Pt()/1000;
    	eta_1 = (*aux_jets_itr_1)->p4().Eta();
    	phi_1 = (*aux_jets_itr_1)->p4().Phi();
    	m_1 = (*aux_jets_itr_1)->p4().M();
    }
    else{
        if(total_eventCounter <=500){
            printf("Event %d filled fake jet 1\n",total_eventCounter);
        }
    }

    // Jet 2
    float D2_2 = -1;
    float subjets_2 = -5;
    float ECF2_2 = -5*TMath::Power(10,9);
    float ECF3_2 = -50*TMath::Power(10,12);
    float pt_2 = -100;
    float eta_2 = -5;
    float phi_2 = -5;
    float m_2 = -10*TMath::Power(10,3);

    if(jet_counter>=2){
    	if((*aux_jets_itr_2)->auxdata<float>("ECF2") != 0){
    		D2_2 = (*aux_jets_itr_2)->auxdata<float>("ECF3");
    		D2_2 = D2_2*pow((*aux_jets_itr_2)->auxdata<float>("ECF1"),3);
    		D2_2 = D2_2/pow((*aux_jets_itr_2)->auxdata<float>("ECF2"),3);
    	}
    	subjets_2 = (float) (*aux_jets_itr_2)->auxdata<int>("NTrimSubjets");
    	ECF2_2 = (*aux_jets_itr_2)->auxdata<float>("ECF2");
    	ECF3_2 = (*aux_jets_itr_2)->auxdata<float>("ECF3");
    	pt_2 = (*aux_jets_itr_2)->p4().Pt()/1000;
    	eta_2 = (*aux_jets_itr_2)->p4().Eta();
    	phi_2 = (*aux_jets_itr_2)->p4().Phi();
    	m_2 = (*aux_jets_itr_2)->p4().M();
    }
    else{
        if(total_eventCounter <=500){
            printf("Event %d filled fake jet 2\n",total_eventCounter);
        }
    }

    // Jet 3
    float D2_3 = -1;
    float subjets_3 = -5;
    float ECF2_3 = -5*TMath::Power(10,9);
    float ECF3_3 = -50*TMath::Power(10,12);
    float pt_3 = -100;
    float eta_3 = -5;
    float phi_3 = -5;
    float m_3 = -10*TMath::Power(10,3);

    if(jet_counter>=3){
    	if((*aux_jets_itr_3)->auxdata<float>("ECF2") != 0){
    		D2_3 = (*aux_jets_itr_3)->auxdata<float>("ECF3");
    		D2_3 = D2_3*pow((*aux_jets_itr_3)->auxdata<float>("ECF1"),3);
    		D2_3 = D2_3/pow((*aux_jets_itr_3)->auxdata<float>("ECF2"),3);
    	}
    	subjets_3 = (float) (*aux_jets_itr_3)->auxdata<int>("NTrimSubjets");
    	ECF2_3 = (*aux_jets_itr_3)->auxdata<float>("ECF2");
    	ECF3_3 = (*aux_jets_itr_3)->auxdata<float>("ECF3");
    	pt_3 = (*aux_jets_itr_3)->p4().Pt()/1000;
    	eta_3 = (*aux_jets_itr_3)->p4().Eta();
    	phi_3 = (*aux_jets_itr_3)->p4().Phi();
    	m_3 = (*aux_jets_itr_3)->p4().M();
    }
    else{
        if(total_eventCounter <=500){
            printf("Event %d filled fake jet 3\n",total_eventCounter);
        }
    }

    if(pt_photon.Pt()/1000>250 && pt_photon.Eta()<1.37 && pt_1>200 && eta_1<2){
    	// baseline cuts

    fprintf(background,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        pt_1,eta_1,phi_1,m_1,D2_1,subjets_1,
        pt_2,eta_2,phi_2,m_2,D2_2,subjets_2,
        pt_3,eta_3,phi_3,m_3,D2_3,subjets_3,
        pt_photon.Pt()/1000,pt_photon.Eta(),pt_photon.Phi(),
        ECF2_1,ECF3_1,ECF2_2,ECF3_2,ECF2_3,ECF3_3,eventInfo->mcEventWeight()
        );
    // fprintf(background,"%f,%f,%f,%f,%f,%d,%f,%f,%f,%f,%f,%d,%f,%f,%f,%f\n",
    //     (*aux_jets_itr_1)->p4().Pt()/1000,(*aux_jets_itr_1)->p4().Eta(),(*aux_jets_itr_1)->p4().Phi(),(*aux_jets_itr_1)->p4().M(),D2_1,(*aux_jets_itr_1)->auxdata<int>("NTrimSubjets"),
    //     (*aux_jets_itr_2)->p4().Pt()/1000,(*aux_jets_itr_2)->p4().Eta(),(*aux_jets_itr_2)->p4().Phi(),(*aux_jets_itr_2)->p4().M(),D2_2,(*aux_jets_itr_2)->auxdata<int>("NTrimSubjets"),
    //     pt_photon.Pt()/1000,pt_photon.Eta(),pt_photon.Phi(),eventInfo->mcEventWeight());

    h0->Fill(pt_photon.Pt()/1000);
    h1->Fill(pt_photon.Eta());
    h2->Fill(pt_photon.Eta(),pt_photon.Pt()/1000);
    h3->Fill(pt_photon.Phi());

    h4->Fill(pt_1);
    h5->Fill(eta_1);
    h6->Fill(phi_1);
    h7->Fill(m_1);
    h8->Fill(ECF2_1);
    h9->Fill(ECF3_1);
    h10->Fill(D2_1);
    h11->Fill(subjets_1);

    h12->Fill(pt_2);
    h13->Fill(eta_2);
    h14->Fill(phi_2);
    h15->Fill(m_2);
    h16->Fill(ECF2_2);
    h17->Fill(ECF3_2);
    h18->Fill(D2_2);
    h19->Fill(subjets_2);

    h20->Fill(pt_3);
    h21->Fill(eta_3);
    h22->Fill(phi_3);
    h23->Fill(m_3);
    h24->Fill(ECF2_3);
    h25->Fill(ECF3_3);
    h26->Fill(D2_3);
    h27->Fill(subjets_3);
}

    if(jet_counter<3 && total_eventCounter<=500){
        printf("Highest Pt jets are:\n (%f, %f, %f, %f, %f, %f)\n(%f, %f, %f, %f, %f, %f)\n(%f, %f, %f, %f, %f, %f)\n",
            pt_1,eta_1,phi_1,m_1,D2_1,subjets_1,
            pt_2,eta_2,phi_2,m_2,D2_2,subjets_2,
            pt_3,eta_3,phi_3,m_3,D2_3,subjets_3
            );
        printf("Highest Pt photon is: (%f,%f,%f)\n\n",pt_photon.Pt()/1000,pt_photon.Eta(),pt_photon.Phi());
    }

    return EL::StatusCode::SUCCESS;
}



EL::StatusCode MyxAODAnalysis :: postExecute ()
{
    // Here you do everything that needs to be done after the main event
    // processing.  This is typically very rare, particularly in user
    // code.  It is mainly used in implementing the NTupleSvc.
	return EL::StatusCode::SUCCESS;
}



EL::StatusCode MyxAODAnalysis :: finalize ()
{
    // This method is the mirror image of initialize(), meaning it gets
    // called after the last event has been processed on the worker node
    // and allows you to finish up any objects you created in
    // initialize() before they are written to disk.  This is actually
    // fairly rare, since this happens separately for each worker node.
    // Most of the time you want to do your post-processing on the
    // submission node after all your histogram outputs have been
    // merged.  This is different from histFinalize() in that it only
    // gets called on worker nodes that processed input events.
	printf("Found %d events\n",total_eventCounter);
    ANA_CHECK_SET_TYPE (EL::StatusCode); // set type of return code you are expecting (add to top of each function once)
    xAOD::TEvent* event = wk()->xaodEvent();
    return EL::StatusCode::SUCCESS;
}



EL::StatusCode MyxAODAnalysis :: histFinalize ()
{
    // This method is the mirror image of histInitialize(), meaning it
    // gets called after the last event has been processed on the worker
    // node and allows you to finish up any objects you created in
    // histInitialize() before they are written to disk.  This is
    // actually fairly rare, since this happens separately for each
    // worker node.  Most of the time you want to do your
    // post-processing on the submission node after all your histogram
    // outputs have been merged.  This is different from finalize() in
    // that it gets called on all worker nodes regardless of whether
    // they processed input events.
	return EL::StatusCode::SUCCESS;
}
