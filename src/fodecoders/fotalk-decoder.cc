//home/huanghe/workspace/keyword-decoder/src/decoder.cc
//
//Author: Pro.Dogegg(Heqing Huang)
//Time:   2017-10-24 16:51:33
//
//
//Function:
//Usage:

#include "keywords-decoder.h"

int main(int argc, char** argv){
    fotalk::KeywordsDecoderConfig config;
    fotalk::KeywordsDecoder keywords_decoder(config);

    std::string model_filename =argv[1],
                fst_filename = argv[2],
                wave_filename = argv[3],
                word_syms_filename = argv[4];

    std::vector<int32> alignment, words;
    //When deocde with weight output 
    kaldi::LatticeWeight weight;

    fst::SymbolTable *word_syms = NULL;
    word_syms = fst::SymbolTable::ReadText(word_syms_filename);

    keywords_decoder.ReadModel(model_filename, fst_filename, word_syms_filename);
    
    //keywords_decoder.Decode(wave_filename, &alignment, &words, "pcm");
    //Decode with weight output
    keywords_decoder.Decode(wave_filename, &alignment, &words, &weight, "pcm");

    std::cout << "Recognize Result: ";
    for( size_t i = 0; i < words.size(); i++ ) {
        std::string s = word_syms->Find(words[i]);
        std::cout << s << ' ';
    }
    std::cout << std::endl;
    std::cout << "Decode Weight: "
              << "Value1: " << weight.Value1()
              << "    "
              << "Value2: " << weight.Value2()
              << std::endl;
}
