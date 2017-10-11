//cvte-decoder/src/online-cvte-decoder.cc
//
//Author:   Pro.Dogegg(Heqing Huang)
//Time:     2017-9-28 09:15:35
//Copyright 2017 SpeakIn
//
//
//Function: 
//Usage: 

#include "transform/cmvn.h"
#include "feat/feature-fbank.h"
#include "feat/wave-reader.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "base/timer.h"
#include "math.h"
//Function: Decoding and cout the result
//@para1: Reference of decoder class
//@para2: Reference of decodable interface class
//@para3: Table of words.txt
bool SpeakinUtteranceDecoder(kaldi::LatticeFasterDecoder& decoder,
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

//Funtion: Computer fbank feature of sigle .wav file
//para1: Path of wav file
//para2: The feature compter result
bool SpeakinFeatureComputer(std::string wave_filename,
                            kaldi::Matrix<kaldi::BaseFloat>* features) {

    kaldi::FbankOptions fbank_opts;
    fbank_opts.mel_opts.num_bins = 40;
    kaldi::BaseFloat vtln_warp_local = 1.0;

    //Read wav file with WaveData
    kaldi::Input wave_input(wave_filename);
    kaldi::WaveData wave_data;
    wave_data.Read(wave_input.Stream());
    kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);

    //Compute Fbank feature
    kaldi::Fbank fbank(fbank_opts);

    try{
        fbank.ComputeFeatures(data, wave_data.SampFreq(), vtln_warp_local, features);  
    } catch (...) {
        std::cout << "Failed to compute features for utterance!" << std::endl;
        return false;
    }
    return true;
}

//Function:Computer feature cmvn
//para1: Input features matrix
//para2: cmvn state matrix
bool SpeakinCmvnComputer(kaldi::Matrix<kaldi::BaseFloat> features,
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
bool SpeakinApplyCmvn(kaldi::Matrix<double> cmvn,
                      kaldi::Matrix<kaldi::BaseFloat> *cmvn_features) {
    try{
        kaldi::ApplyCmvn(cmvn, false, cmvn_features);
    } catch(...) {
        std::cout << "Apply cmvn error!" << std::endl;
        return false;
    }
    return true;
}

//Function:
//para1:
//para2:
bool SpeakinWordsLocation(std::vector<int32> ali,
        std::vector<int32>* locations) {
    int32 phone_start = ali[0];
    locations->push_back(0);

    for(size_t i = 1; i < ali.size(); i++) {
        if(ali[i] != phone_start) {
            if(std::abs(ali[i] - phone_start) != 1) {
                locations->push_back(i);
                phone_start = ali[i];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    //using namespace kaldi;
    //using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    //using fst::SymbolTable;
    //using fst::Fst;
    //using fst::StdArc;
    
    //Init decoding configures
    kaldi::LatticeFasterDecoderConfig config;
    config.max_active = 7000;
    config.lattice_beam = 8.0;
    config.beam = 15.0;
    //Init nnet compute configures  
    kaldi::nnet3::NnetSimpleComputationOptions decodable_opts;
    decodable_opts.acoustic_scale = 1.0;

    int32 online_ivector_period = 0;
    std::string ivector_rspecifier,
                online_ivector_rspecifier;

    //Read options
    std::string model_in_filename = argv[1],
                fst_in_str =argv[2],
                wave_filename = argv[3],
                word_syms_filename = argv[4];
   
    //Read the final.mdl
    kaldi::TransitionModel trans_model;
    kaldi::nnet3::AmNnetSimple am_nnet;
    {
        bool binary;
        kaldi::Input ki(model_in_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_nnet.Read(ki.Stream(), binary);
        SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
        SetDropoutTestMode(true, &(am_nnet.GetNnet()));
        CollapseModel(kaldi::nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    //Init decode compiler
    //bool determinize = config.determinize_lattice;
    kaldi::nnet3::CachingOptimizingCompiler compiler(am_nnet.GetNnet(),
                                                     decodable_opts.optimize_config);

    //Read HLCG.fst
    fst::Fst<fst::StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
    fst::SymbolTable *word_syms = NULL;
    word_syms = fst::SymbolTable::ReadText(word_syms_filename);

    {
        const kaldi::Matrix<kaldi::BaseFloat> *online_ivectors = NULL;
        const kaldi::Vector<kaldi::BaseFloat> *ivector = NULL;
        kaldi::Matrix<kaldi::BaseFloat> features;
        kaldi::Matrix<double> cmvn;
       
        //Compute fband 
        SpeakinFeatureComputer(wave_filename, &features);
        //Compute cmvn
        SpeakinCmvnComputer(features,&cmvn);
        //Apply cmvn to feature
        SpeakinApplyCmvn(cmvn, &features);
        //TODO init online_features ivectors and ivector(is null here)
        
        //Init deocder
        kaldi::LatticeFasterDecoder decoder(*decode_fst, config);
        kaldi::nnet3::DecodableAmNnetSimple nnet_decodable(decodable_opts, trans_model,
                                                           am_nnet, features, ivector,
                                                           online_ivectors,
                                                           online_ivector_period,
                                                           &compiler);
        //Decoding
        std::vector<int32> alignment,words;
        SpeakinUtteranceDecoder(decoder, nnet_decodable, &alignment, &words);

        //Compute split location
        std::vector<int32> locations;
        SpeakinWordsLocation(alignment, &locations);
        //cout the result
        std::cout << "RECOGNIZE RESULT: ";
        for( size_t i = 0; i < words.size(); i++ ) {
            std::string s = word_syms->Find(words[i]);
            std::cout << s << ' ';
        }
        std::cout << std::endl;

        std::cout << "ALIGNMENT RESULT: ";
        for( size_t i = 0; i < alignment.size(); i++ ) {
            std::cout << alignment[i] << ' ';
        }
        std::cout << std::endl;
        
        std::cout << "LOCATION RESULT: ";
        for( size_t i = 0; i < locations.size(); i++ ) {
            std::cout << locations[i] << ' ';
        }
        std::cout << std::endl;
        
        std::cout << "LOCATION START: ";
        for( size_t i = 0; i < locations.size(); i++ ) {
            std::cout << alignment[locations[i]] << ' ';
        }
        std::cout << std::endl;
    }
    
}//main

