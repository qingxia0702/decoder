//cvte-decoder/src/decoder.cc
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
    
    speakin::DigiterDecoderConfig config;
    speakin::DigiterDecoder digiter_decoder(config);
    
    speakin::SplitOptions option;
    speakin::Spliter wave_spliter(option, false);

    //Read options
    std::string model_filename = argv[1],
                fst_filename =argv[2],
                wave_filename = argv[3],
                word_syms_filename = argv[4];

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
    std::cout << "ALIGNMENT RESULT: ";
    for( size_t i = 0; i < alignment.size(); i++ ) {
        std::cout << alignment[i] << ' ';
    }
    std::cout << std::endl;

    wave_spliter.SilenceLocation(alignment);
}
