#include <TFile.h>
#include <TH1.h>
#include "TF1.h"
#include "TCanvas.h"
#include "TStyle.h"


void save(){
    
    std::string input = "./Run/wqq280/hist-mc15_13TeV.305436.Sherpa_CT10_WqqGammaPt280_500.merge.DAOD_EXOT3.e5037_s2726_r7772_r7676_p2949.root";
    TFile *f0 = new TFile(input.c_str());
    
    int i=0;
    
    for(i=0;i<=28;i++){
        char name[50];
        sprintf(name,"h%d",i);
        printf("Histogram : %s\n",name);
        TH1F *h = (TH1F*) f0->Get(name);
        
        // Save the histogram as a .png file
        TCanvas *c = new TCanvas;
        h->Draw("HIST");
        c->Update();
        sprintf(name,"./Signal Hist/h%d.png",i);
        c->SaveAs(name);
    }

    f0->Close();

    input = "./Run/gjets280/hist-mc15_13TeV.361048.Sherpa_CT10_SinglePhotonPt280_500_CVetoBVeto.merge.DAOD_EXOT3.e3587_s2608_s2183_r7725_r7676_p2949.root";
    TFile *f1 = new TFile(input.c_str());
    
    i=0;
    
    for(i=0;i<=27;i++){
        char name[50];
        sprintf(name,"h%d",i);
        printf("Histogram : %s\n",name);
        TH1F *h = (TH1F*) f1->Get(name);
        
        // Save the histogram as a .png file
        TCanvas *c = new TCanvas;
        h->Draw("HIST");
        c->Update();
        sprintf(name,"./Background Hist/h%d.png",i);
        c->SaveAs(name);
    }
    f1->Close();

    // std::string input = "./Run/wqq140/hist-mc15_13TeV.305435.Sherpa_CT10_WqqGammaPt140_280.merge.DAOD_EXOT3.e5037_s2726_r7772_r7676_p2949.root";
    // TFile *f0 = new TFile(input.c_str());
    
    // int i=0;
    
    // for(i=0;i<=27;i++){
    //     char name[50];
    //     sprintf(name,"h%d",i);
    //     printf("Histogram : %s\n",name);
    //     TH1F *h = (TH1F*) f0->Get(name);
        
    //     // Save the histogram as a .png file
    //     TCanvas *c = new TCanvas;
    //     h->Draw("HIST");
    //     c->Update();
    //     sprintf(name,"./wqq 140 Signal Hist/h%d.png",i);
    //     c->SaveAs(name);
    // }

}
