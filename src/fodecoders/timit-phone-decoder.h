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
#include "matrix/matrix-common.h"
#include "transform/cmvn.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "feat/feature-functions.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "nnet2/decodable-am-nnet.h"
#include "fstext/kaldi-fst-io.h"
#include "pcm-reader.h"
namespace fotalk {

struct TimitPhoneDecoderConfig{
    kaldi::LatticeFasterDecoderConfig decoder_config;
    kaldi::BaseFloat acoustic_scale;
    kaldi::MfccOptions mfcc_opts;
    kaldi::BaseFloat vtln_warp;
    kaldi::DeltaFeaturesOptions delta_opts;
    bool pad_input;
    int32 left_context;
    int32 right_context;

    TimitPhoneDecoderConfig(){
        //Change default value of initialization structors
        decoder_config.max_active = 7000;
        decoder_config.lattice_beam = 8.0;
        decoder_config.beam = 15.0;

        mfcc_opts.use_energy = false;
        acoustic_scale = 0.1;
        vtln_warp = 1.0;
        pad_input = true;

        left_context = 3;
        right_context = 3;
    }
};

class TimitPhoneDecoder {
  private:
    TimitPhoneDecoderConfig config_;
    kaldi::TransitionModel trans_model_;
    kaldi::nnet2::AmNnet am_nnet_;
    fst::Fst<fst::StdArc>* decode_fst_;
    fst::SymbolTable* word_syms_;
    kaldi::Matrix<kaldi::BaseFloat> global_transform_;

  public:
    TimitPhoneDecoder(TimitPhoneDecoderConfig config){
       config_ = config; 
    }

    //Read model
    bool ReadModel(std::string model_filename,
                   std::string fst_filename,
                   std::string word_syms_filename,
                   std::string mat_filename){
        bool binary;
        //Read final.mdl
        try {
            kaldi::Input ki(model_filename, &binary);
            trans_model_.Read(ki.Stream(), binary);
            am_nnet_.Read(ki.Stream(), binary);
        } catch (...) {
            std::cout << "ERROR: Read model file " 
                      << model_filename << "failed!" << std::endl;
        }

        //Read HCLG.fst
        decode_fst_ = fst::ReadFstKaldiGeneric(fst_filename);

        //Read final.mat
        kaldi::ReadKaldiObject(mat_filename, &global_transform_);    

        //Read words.txt
        word_syms_ = fst::SymbolTable::ReadText(word_syms_filename);
    }
    //Function: Computer mfcc features of a single .pcm file
    //@para1: The vector generate by PcmReader 
    //@para2: Output the features matrix
    bool FeatureComputer(kaldi::Vector<kaldi::BaseFloat> pcm_vec,
                         kaldi::Matrix<kaldi::BaseFloat>* features){
        kaldi::Mfcc mfcc(config_.mfcc_opts);
        try{
            mfcc.Compute(pcm_vec, config_.vtln_warp, features);
            //mfcc.ComputeFeatures(pcm_vec, samp_freq, config_.vtln_warp, features); 
        } catch (...) {
            std::cout << "Failed to compute features for utterance!" << std::endl;
            return false;
        }
        return true;
    }

    //Funtion: Computer mfcc feature of sigle .wav file
    //para1: Path of wav file
    //para2: The feature compter result
    bool FeatureComputer(std::string wave_filename,
                         kaldi::Matrix<kaldi::BaseFloat>* features) {
        //Read wav file with WaveData
        kaldi::Input wave_input(wave_filename);
        kaldi::WaveData wave_data;
        wave_data.Read(wave_input.Stream());
        kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);
        //Compute mfcc feature
        kaldi::Mfcc mfcc(config_.mfcc_opts);
        try{
            mfcc.Compute(data, config_.vtln_warp, features);
            //mfcc.ComputeFeatures(data, wave_data.SampFreq(), config_.vtln_warp, features); 
        } catch (...) {
            std::cout << "Failed to compute features for utterance!" << std::endl;
            return false;
        }
        return true;
    }
    //Function: Do add deltas to features matrix
    //para1: Input features
    //para2: Output the new features
    bool AddDeltas(kaldi::Matrix<kaldi::BaseFloat> features,
                    kaldi::Matrix<kaldi::BaseFloat>* new_features){
        try{
            kaldi::ComputeDeltas(config_.delta_opts, features, new_features);
        } catch(...){
            std::cout << "Add Deltas error!" << std::endl;
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
    //Function:
    //@para1:
    //@para2:
    void TransformFeatures(kaldi::Matrix<kaldi::BaseFloat> features,
            kaldi::Matrix<kaldi::BaseFloat>* new_features){
        std::cout << "The Dim of final.mat is: " 
                  << "Col: " << global_transform_.NumCols()
                  << " Row: " << global_transform_.NumRows() << std::endl;
        std::cout << "The Dim of features is: " 
                  << "Col: " << features.NumCols()
                  << " Row: " << features.NumRows() << std::endl;

        new_features->Resize(features.NumRows(), global_transform_.NumRows());
        const kaldi::Matrix<kaldi::BaseFloat> &trans = global_transform_;
        if(trans.NumCols() == features.NumCols()){
            new_features->AddMatMat(1.0, features, kaldi::kNoTrans, trans, kaldi::kTrans, 0.0);
        } else if(trans.NumCols() == features.NumCols() + 1){
            kaldi::SubMatrix<kaldi::BaseFloat> linear_part(trans, 0, trans.NumRows(), 
                                                           0, features.NumCols());
            new_features->AddMatMat(1.0, features, kaldi::kNoTrans, linear_part, kaldi::kTrans, 0.0);
            kaldi::Vector<kaldi::BaseFloat> offset(trans.NumRows());
            offset.CopyColFromMat(trans, features.NumCols());
            new_features->AddVecToRows(1.0, offset);
        }
        std::cout << "The Dim of transformed features is: " 
                  << "Col: " << new_features->NumCols()
                  << " Row: " << new_features->NumRows() << std::endl;
    }
    //Function:
    //@para1:
    //@para2:
    void SpliceFeatures(kaldi::Matrix<kaldi::BaseFloat> features,
            kaldi::Matrix<kaldi::BaseFloat>* new_features){
        kaldi::SpliceFrames(features, config_.left_context, config_.right_context, new_features);
        std::cout << "The Dim of Spliced features is: " 
                  << "Col: " << new_features->NumCols()
                  << " Row: " << new_features->NumRows() << std::endl;
    } 
    
    //Function: Decoding and cout the result
    //@para1: Reference of decoder class
    //@para2: Reference of decodable interface class
    //@para3: Output of alignmet vector
    //@para3: Output of words vector
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
    //Function: Decoding and cout the result
    //@para1: Reference of decoder class
    //@para2: Reference of decodable interface class
    //@para3: Output of alignmet vector
    //@para4: Output of words vector
    //@para5: Output of lattice weights
    bool UtteranceDecoder(kaldi::LatticeFasterDecoder& decoder,
                          kaldi::DecodableInterface& decodable,
                          std::vector<int32>* alignment,
                          std::vector<int32>* words,
                          kaldi::LatticeWeight* weight) {

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

        fst::GetLinearSymbolSequence(decoded, alignment, words, weight);
        
        return true;
    }
    //Function:Do decoder operation of input wave file,Decode without output wegit use this method 
    //@para1: Path of wave to decode
    //@para2: Output of alignment
    //@para3: Output of words
    //@para4: Format of input file,default is .wav
    bool Decode(std::string wave_filename,
                std::vector<int32>* alignment,
                std::vector<int32>* words,
                std::string file_format="wav"){

        kaldi::Matrix<kaldi::BaseFloat> features, delta_features, decoder_features,
                                        trans_features, slice_features;
        kaldi::Matrix<double> cmvn;
       
        if (file_format == "wav"){
            FeatureComputer(wave_filename, &features);
        }else if (file_format == "pcm"){
            kaldi::Input wave_input(wave_filename);
            fotalk::PcmReaderConfig pcm_config;
            fotalk::PcmReader pcm_reader(pcm_config);
            pcm_reader.Read(wave_input.Stream());
            //kaldi::SubVector<kaldi::BaseFloat> temp(pcm_reader.Data(), 0);
            //kaldi::Vector<kaldi::BaseFloat> pcm_vec(temp.Dim());
            //pcm_vec.CopyFromVec(temp);
            kaldi::Vector<kaldi::BaseFloat> pcm_vec(pcm_reader.Data().NumCols());
            pcm_vec.CopyRowFromMat(pcm_reader.Data(), 0);
            FeatureComputer(pcm_vec, &features);
        }

        CmvnComputer(features,&cmvn); 
        ApplyCmvn(cmvn, &features);
        SpliceFeatures(features, &slice_features);
        TransformFeatures(slice_features, &decoder_features);
        
        kaldi::CuMatrix<kaldi::BaseFloat> cu_delta_features(decoder_features.NumRows(), decoder_features.NumCols());
        cu_delta_features.CopyFromMat(decoder_features);

        kaldi::nnet2::DecodableAmNnet nnet_decodable(trans_model_, am_nnet_,
                                                     cu_delta_features, config_.pad_input, config_.acoustic_scale);
        kaldi::LatticeFasterDecoder decoder(*decode_fst_, config_.decoder_config);
        UtteranceDecoder(decoder, nnet_decodable, alignment, words);
    }
    //Function:Do decoder operation of input wave file,Decode with output wegit use this method 
    //@para1: Path of wave to decode
    //@para2: Output of alignment
    //@para3: Output of words
    //@para4: Output of weight
    //@para5: Format of input file,default is .wav
    bool Decode(std::string wave_filename,
                std::vector<int32>* alignment,
                std::vector<int32>* words,
                kaldi::LatticeWeight* weight,
                std::string file_format="wav"){

        kaldi::Matrix<kaldi::BaseFloat> features, delta_features, decoder_features,
                                        trans_features, slice_features;
        kaldi::Matrix<double> cmvn;
       
        if (file_format == "wav"){
            FeatureComputer(wave_filename, &features);
        }else if (file_format == "pcm"){
            kaldi::Input wave_input(wave_filename);
            fotalk::PcmReaderConfig pcm_config;
            fotalk::PcmReader pcm_reader(pcm_config);
            pcm_reader.Read(wave_input.Stream());
            //kaldi::SubVector<kaldi::BaseFloat> temp(pcm_reader.Data(), 0);
            //kaldi::Vector<kaldi::BaseFloat> pcm_vec(temp.Dim());
            //pcm_vec.CopyFromVec(temp);
            kaldi::Vector<kaldi::BaseFloat> pcm_vec(pcm_reader.Data().NumCols());
            pcm_vec.CopyRowFromMat(pcm_reader.Data(), 0);
            FeatureComputer(pcm_vec, &features);
        }

        CmvnComputer(features,&cmvn); 
        ApplyCmvn(cmvn, &features);
        SpliceFeatures(features, &slice_features);
        TransformFeatures(slice_features, &decoder_features);
        
        kaldi::CuMatrix<kaldi::BaseFloat> cu_delta_features(decoder_features.NumRows(), decoder_features.NumCols());
        cu_delta_features.CopyFromMat(decoder_features);
        kaldi::LatticeFasterDecoder decoder(*decode_fst_, config_.decoder_config);
        kaldi::nnet2::DecodableAmNnet nnet_decodable(trans_model_, am_nnet_,
                                                     cu_delta_features, config_.pad_input, config_.acoustic_scale);
        UtteranceDecoder(decoder, nnet_decodable, alignment, words, weight);
    }
    //Function:Do decoder operation of input a data buffer
    //@para1: Input data buffer
    //@para2: Input buffer size
    //@para3: Output of words
    //@para4: Output of weight
    //@para5: Format of input file,default is .wav
    bool Decode(char* data,
                int data_len,
                std::vector<int32>* alignment,
                std::vector<int32>* words,
                kaldi::LatticeWeight* weight){

        kaldi::Matrix<kaldi::BaseFloat> features, delta_features, decoder_features,
                                        trans_features, slice_features;
        kaldi::Matrix<double> cmvn;
        std::string str(data, data + data_len);
        std::stringstream ss(str);

        fotalk::PcmReaderConfig pcm_config;
        fotalk::PcmReader pcm_reader(pcm_config);
        pcm_reader.Read(ss);
        //kaldi::SubVector<kaldi::BaseFloat> temp(pcm_reader.Data(), 0);
        //kaldi::Vector<kaldi::BaseFloat> pcm_vec(temp.Dim());
        //pcm_vec.CopyFromVec(temp);
        kaldi::Vector<kaldi::BaseFloat> pcm_vec(pcm_reader.Data().NumCols());
        pcm_vec.CopyRowFromMat(pcm_reader.Data(), 0);
        
        CmvnComputer(features,&cmvn); 
        ApplyCmvn(cmvn, &features);
        SpliceFeatures(features, &slice_features);
        TransformFeatures(slice_features, &decoder_features);
        
        kaldi::CuMatrix<kaldi::BaseFloat> cu_delta_features(decoder_features.NumRows(), decoder_features.NumCols());
        cu_delta_features.CopyFromMat(decoder_features);
        kaldi::LatticeFasterDecoder decoder(*decode_fst_, config_.decoder_config);
        kaldi::nnet2::DecodableAmNnet nnet_decodable(trans_model_, am_nnet_,
                                                     cu_delta_features, config_.pad_input, config_.acoustic_scale);
        UtteranceDecoder(decoder, nnet_decodable, alignment, words, weight);
    }
    //Function:
    //@para1:
    //@para2:
    bool AliToPhone(std::vector<int32> alignment,
            std::vector<int32>* phone_id_list){
        std::vector<std::vector<int32> > split;
        kaldi::SplitToPhones(trans_model_, alignment, &split);
        for (size_t i =0; i < split.size(); i++){
           int32 phone = trans_model_.TransitionIdToPhone(split[i][0]);
           int32 num_repeats = split[i].size();
           for(int32 j = 0; j< num_repeats; j++)
               phone_id_list->push_back(phone);
        }
    }
};
}//namespace _fotalk
