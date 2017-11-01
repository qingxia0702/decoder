//cvte-decoder/src/word_split.cc
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

int main(int argc, char** argv){

    const char* usage = 
        "Usage: \n "
        "    ./word_splite final.mdl HCLG.fst *.wav words.txt $num_of_word_pre_utt $wave_output_path\n"
        "word_num_pre_utt mean how many words you want in a splited utterance";

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
                wave_filename = argv[3],
                word_syms_filename = argv[4],
                wave_output_path = argv[6];
    
    speakin::SplitOptions option;
    option.word_num_pre_utt = std::atoi(argv[5]);
    speakin::Spliter wave_spliter(option, true);

    std::vector<int32> alignment, words;
    fst::SymbolTable *word_syms = NULL;
    word_syms = fst::SymbolTable::ReadText(word_syms_filename);

    digiter_decoder.ReadModel(model_filename, fst_filename, word_syms_filename);
    digiter_decoder.Decode(wave_filename, &alignment, &words);
    //cout the result
    std::cout << "RECOGNIZE RESULT: ";
    for( size_t i = 0; i < words.size(); i++ ) {
        std::string s = word_syms->Find(words[i]);
        std::cout << s << ' ';
    }
    std::cout << std::endl;
    //std::cout << "ALIGNMENT RESULT: ";
    //for( size_t i = 0; i < alignment.size(); i++ ) {
        //std::cout << alignment[i] << ' ';
    //}
    //std::cout << std::endl;

    wave_spliter.WordsLocation(alignment);
    wave_spliter.WavSpliter(config.fbank_opts.frame_opts.frame_length_ms/1000,
                            config.fbank_opts.frame_opts.frame_shift_ms/1000,
                            wave_filename,
                            wave_output_path);
}
