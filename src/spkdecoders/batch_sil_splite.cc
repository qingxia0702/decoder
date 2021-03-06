//cvte-decoder/src/sil_split.cc
//
//Author:   Pro.Dogegg(Heqing Huang)
//Time:     2017-10-18 09:15:35
//Copyright 2017 SpeakIn
//
//
//Function: 
//Usage:

#include "digiter-decoder.h"
#include "split-util.h"
#include <iostream>

int main(int argc, char** argv){
    
    const char* usage = 
        "Usage: \n "
        "    ./sil_splite final.mdl HCLG.fst *.wav words.txt $num_min_slience $wave_out_put_path\n"
        "num_mini_slience mean the mininize frame of silence you want to split";

    if(argc != 6){
        std::cout << "ERROR: Wrong options!" << std::endl;
        std::cout << usage << std::endl;
        return -1;
    }

    speakin::DigiterDecoderConfig config;  
    speakin::DigiterDecoder digiter_decoder(config);

    //Read options
    std::string model_filename = argv[1],
                fst_filename =argv[2],
                wave_filename_scp = argv[3],
                word_syms_filename = argv[4];
    
    int32 min_silence = std::atoi(argv[5]);
                
    speakin::SplitOptions option;
    option.min_silence = min_silence;
    speakin::Spliter wave_spliter(option, true);

    std::vector<int32> alignment, words;
    fst::SymbolTable *word_syms = NULL;
    word_syms = fst::SymbolTable::ReadText(word_syms_filename);

    //kaldi::RandomAccessTableReader<std::string> wave_file_reader(wave_filename_scp);
    kaldi::SequentialTokenReader wave_file_reader(wave_filename_scp);

    digiter_decoder.ReadModel(model_filename, fst_filename, word_syms_filename);
    

    for(; !wave_file_reader.Done(); wave_file_reader.Next()){
        std::string wave_output_path = wave_file_reader.Value();
        std::string wave_filename = wave_file_reader.Key();
        std::cout << "(LOG)wave output path: " << wave_output_path << std::endl;
        std::cout << "(LOG)wave input path: " << wave_filename << std::endl;
        
        digiter_decoder.Decode(wave_filename, &alignment, &words);
        
        std::cout << "(LOG)RECOGNIZE RESULT: ";
        for( size_t i = 0; i < words.size(); i++ ) {
            std::string s = word_syms->Find(words[i]);
            std::cout << s << ' ';
        }
        std::cout << std::endl;
        
        wave_spliter.SilenceLocation(alignment);
        wave_spliter.WavSpliter(config.fbank_opts.frame_opts.frame_length_ms/1000,
                                config.fbank_opts.frame_opts.frame_shift_ms/1000,
                                wave_filename,
                                wave_output_path);
    }
        //std::cout << "ALIGNMENT RESULT: ";
    //for( size_t i = 0; i < alignment.size(); i++ ) {
        //std::cout << alignment[i] << ' ';
    //}
    //std::cout << std::endl;
}
