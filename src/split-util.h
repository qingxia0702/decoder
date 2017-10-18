//cvte-decoder/src/split-util.h
//
//Author: Pro.Dogegg(Heqing Huang)
//Time: 2017-10-17 16:44:11
//Copyright 2017 SpeakIn
//
//
//Function:
//Usage:


#ifndef _SPLIT_UTIL_H
#define _SPLIT_UTIL_H
#endif
#include "base/kaldi-common.h"
#include "feat/wave-reader.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace speakin{

struct SplitOptions {
    //Value for split by silence
    int32 min_silence;
    int32 max_split;
    int32 min_frame_num;
    //Value for split file by a constant number
    int32 phone_num_pre_word;
    int32 word_num_pre_utt;
    int32 silence_pre_id;
    int32 silence_id;
    
    SplitOptions() : min_silence(15),
                     max_split(10),
                     min_frame_num(50),
                     phone_num_pre_word(2),
                     word_num_pre_utt(8),
                     silence_pre_id(2),
                     silence_id(1){};
};

class Spliter {
  private:
    SplitOptions options_;
    std::vector<int32> locations_;
    int32 num_;
    bool silence_spliter_;

  public:
    Spliter(SplitOptions opt, bool flag){
        options_ = opt;
        silence_spliter_ = flag;
    }
    

    int32 ComputeSplitPhoneNum(){
        num_ = options_.word_num_pre_utt * options_.phone_num_pre_word;
        return num_;
    }
    
    //Function: Find the location to splite depended on words
    //para1: Alignment vector
    void WordsLocation(std::vector<int32> ali){
        int32 phone_start = ali[0];
        int32 index = 1;
        for(size_t i = 0; i < ali.size(); i++) {
            if(ali[i] != phone_start) {
                if(std::abs(ali[i] - phone_start) != 1) {
                    if(ali[i] != options_.silence_pre_id) {
                        if(index == ComputeSplitPhoneNum()) {
                            locations_.push_back(i);
                            index = 1;
                        } else {
                            index++;
                        }
                    }
                    phone_start = ali[i];
                }
            }
        }
        std::cout << "LOCATION RESULT: ";
        for( size_t i = 0; i < locations_.size(); i++ ) {
            std::cout << locations_[i] << ' ';
        }
        std::cout << std::endl;
        std::cout << "LOCATION STATE: ";
        for( size_t i = 0; i < locations_.size(); i++ ) {
            std::cout << ali[locations_[i]] << ' ';
        }
        std::cout << std::endl;
    }

    //Function: Find the location to splite depended on silence
    //para1: Alignment vector
    void SilenceLocation(std::vector<int32> ali){
        std::map<int32, std::vector<int32> > silence_info_map;
        std::vector<int32> info;
        int32 sil_start = 0, sil_end = 0, sil_lenth = 0;
        bool sil_flag = false;

        //Find all silence and it lenth
        for(size_t i = 0; i < ali.size(); i++){
            if(ali[i] == options_.silence_pre_id) {
                sil_flag = true;
                sil_start = i;
                sil_lenth++;
                info.push_back(sil_start);
            } else if(ali[i] == options_.silence_id){
                sil_lenth++;
            } else {
                //sil_end = i - 1;
                if(sil_flag){
                    info.push_back(i - 1);
                    silence_info_map.insert(std::pair<int32, std::vector<int32> >(sil_lenth, info));
                    sil_flag = false;
                    sil_end = 0;
                    sil_start = 0;
                    sil_lenth = 0;
                    info.clear();
                }
            }

        }

        //Print silence information
        std::map<int32, std::vector<int32> >::iterator index;
        for(index = silence_info_map.begin(); index != silence_info_map.end(); index++){
            std::cout << "Silence information: " << index->first 
                << " " << index->second[0] << " "  << index->second[1] << std::endl; 
        }
    }

    //Function: Split a long wav file to pieces
    //para2: Frame length used in feature extraction
    //para3: Frame offset used in feature extraction
    //para4: Wave file need to split
    bool WavSpliter(kaldi::BaseFloat frame_len,
                           kaldi::BaseFloat offset,
                           std::string wave_filename){
        
        std::string wave_output_filename;
        if (silence_spliter_){
            wave_output_filename = "sil_";
        }
        
        kaldi::Input wave_input(wave_filename);
        kaldi::WaveData wave_data_input;
        wave_data_input.Read(wave_input.Stream());
        bool binary = true;
        
        std::cout << "wave SampFreq: " << wave_data_input.SampFreq() << std::endl;
        std::cout << "wave data cols: " << wave_data_input.Data().NumCols() << std::endl;
        std::cout << "wave data rows: " << wave_data_input.Data().NumRows() << std::endl;

        int32 splite_start, splite_end;
        //Foreach the input vector to get the loaction to split
        try{
            std::cout << "splite location is:" << std::endl;
            for(size_t i = 0; i < locations_.size() - 1; i++) {
                //Transform frame location to wave data location
                splite_start = static_cast<int32>(locations_[i] * offset * wave_data_input.SampFreq());
                splite_end = static_cast<int32>((locations_[i + 1] * offset + frame_len) * wave_data_input.SampFreq());
                
                std::cout << locations_[i] << " ";
                std::cout << locations_[i + 1] << "------";
                std::cout << splite_start << " ";
                std::cout << splite_end << std::endl;
                //Output file path 
                wave_output_filename += std::to_string(i) + ".wav";
                std::cout << "Output path:" << wave_output_filename << std::endl;
                
                //Get first row of original wave data matrix(channal 1)
                kaldi::SubVector<kaldi::BaseFloat> temp_vec(wave_data_input.Data(), 0);
                //Do split with start and end location
                kaldi::SubVector<kaldi::BaseFloat> splite_vec(temp_vec, splite_start, (splite_end - splite_start + 1));
                kaldi::Matrix<kaldi::BaseFloat> output_data(1 ,splite_vec.Dim());
                output_data.CopyRowFromVec(splite_vec, 0);
                
                //Write new wave file
                kaldi::WaveData wave_data_output(wave_data_input.SampFreq(), output_data);
                kaldi::Output wave_output(wave_output_filename, binary, false);
                wave_data_output.Write(wave_output.Stream());
            }
        } catch(...) {
            std::cout << "ERROR: Do wave splite failed!" << std::endl;
            return false;
        }
        return true;
    }//WaveSpliter()

};//class Spliter
}//namespace speakin
