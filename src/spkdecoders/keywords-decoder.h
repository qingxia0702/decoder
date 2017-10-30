//cvte-decoder/src/digiter-decoder.h
//
//Author:   Pro.Dogegg(Heqing Huang)
//Time:     2017-9-28 09:15:35
//Copyright 2017 SpeakIn
//
//
//Function: 
//Usage: 

#ifndef _DIGITER_DECODE_H
#define _DIGITER_DECODE_H
#endif

#include "transform/cmvn.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "base/timer.h"

namespace fotalk {

struct KeywordsDecoderConfig{
    kaldi::LatticeFasterDecoderConfig decoder_config;
    kaldi::BaseFloat acoustic_scale;
    kaldi::MfccOptions mfcc_opts;
    kaldi::BaseFloat vtln_warp;

    KeywordsDecoderConfig(){
        //Change default value of initialization structors
        decoder_config.max_active = 4000;
        decoder_config.lattice_beam = 8.0;
        decoder_config.beam = 12.0;

        //mfcc_opts.use_energy = false;
        mfcc_opts.num_ceps = 39;
        acoustic_scale = 0.1;
        vtln_warp = 1.0;
    }
};

class KeywordsDecoder {
  private:
    KeywordsDecoderConfig config_;
    kaldi::TransitionModel trans_model_;
    kaldi::AmDiagGmm am_gmm_;
    fst::Fst<fst::StdArc>* decode_fst_;
    fst::SymbolTable* word_syms_;

  public:
    KeywordsDecoder(KeywordsDecoderConfig config){
       config_ = config; 
    }

    //Read model
    bool ReadModel(std::string model_filename,
                   std::string fst_filename,
                   std::string word_syms_filename){
        bool binary;
        //Read final.mdl
        try {
            kaldi::Input ki(model_filename, &binary);
            trans_model_.Read(ki.Stream(), binary);
            am_gmm_.Read(ki.Stream(), binary);
        } catch (...) {
            std::cout << "ERROR: Read model file " 
                      << model_filename << "failed!" << std::endl;
        }

        //Read HCLG.fst
        decode_fst_ = fst::ReadFstKaldiGeneric(fst_filename);

        //Read words.txt
        word_syms_ = fst::SymbolTable::ReadText(word_syms_filename);
    }

    //Funtion: Computer fbank feature of sigle .wav file
    //para1: Path of wav file
    //para2: The feature compter result
    bool FeatureComputer(std::string wave_filename,
                                kaldi::Matrix<kaldi::BaseFloat>* features) {
        //Read wav file with WaveData
        kaldi::Input wave_input(wave_filename);
        kaldi::WaveData wave_data;
        wave_data.Read(wave_input.Stream());
        kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);

        //Compute Fbank feature
        kaldi::Mfcc mfcc(config_.mfcc_opts);
        try{
            mfcc.ComputeFeatures(data, wave_data.SampFreq(), config_.vtln_warp, features); 
        } catch (...) {
            std::cout << "Failed to compute features for utterance!" << std::endl;
            return false;
        }
        return true;
    }
    
    //Function:Computer feature cmvn
    //para1: Input features matrix
    //para2: cmvn state matrix
    bool CmvnComputer(kaldi::Matrix<kaldi::BaseFloat> features,
                             kaldi::Matrix<double>* cmvn) {
        try{
            kaldi::InitCmvnStats(features.NumCols(), cmvn);
            kaldi::AccCmvnStats(features, NULL, cmvn);
        } catch(...) {
            std::cout << "Compute cmvn error!" << std::endl;
            return false;
        }
        return true;
    }

    //Function:Apply cmvn to feature matrix
    //para1: Cmvn states to apply
    //para2: feature matrix with cmvn
    bool ApplyCmvn(kaldi::Matrix<double> cmvn,
                          kaldi::Matrix<kaldi::BaseFloat> *cmvn_features) {
        try{
            kaldi::ApplyCmvn(cmvn, false, cmvn_features);
        } catch(...) {
            std::cout << "Apply cmvn error!" << std::endl;
            return false;
        }
        return true;
    }
    
    //Function: Decoding and cout the result
    //@para1: Reference of decoder class
    //@para2: Reference of decodable interface class
    //@para3: Table of words.txt
    bool UtteranceDecoder(kaldi::LatticeFasterDecoder& decoder,
                                 kaldi::DecodableInterface& decodable,
                                std::vector<int32>* alignment,
                                std::vector<int32>* words) {

        if(!decoder.Decode(&decodable)) {
            std::cout << "Failed to decode file!" << std::endl;
            return false;
        } 

        if(!decoder.ReachedFinal()) {
            std::cout << "Not producing output for utterance!" << std::endl;
            return false;
        }
        
        fst::VectorFst<kaldi::LatticeArc>  decoded;
        if(!decoder.GetBestPath(&decoded)) {
            std::cout << "Failed to get traceback for utterance " << std::endl;
        }

        kaldi::LatticeWeight weight;
        fst::GetLinearSymbolSequence(decoded, alignment, words, &weight);
        
        return true;
    }
    //Initialization of decoder by input models
    bool Decode(std::string wave_filename,
                std::vector<int32>* alignment,
                std::vector<int32>* words){
       
        kaldi::Matrix<kaldi::BaseFloat> features;
        kaldi::Matrix<double> cmvn;
        
        FeatureComputer(wave_filename, &features);
        CmvnComputer(features,&cmvn); 
        ApplyCmvn(cmvn, &features);

        kaldi::LatticeFasterDecoder decoder(*decode_fst_, config_.decoder_config);
        kaldi::DecodableAmDiagGmmScaled gmm_decodable(am_gmm_, trans_model_, features,
                                              config_.acoustic_scale);
        UtteranceDecoder(decoder, gmm_decodable, alignment, words);
    }
};
}//namespace fotalk
