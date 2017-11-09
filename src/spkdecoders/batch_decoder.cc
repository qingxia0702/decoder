//home/huanghe/workspace/keyword-decoder/src/decoder.cc
//
//Author: Pro.Dogegg(Heqing Huang)
//Time:   2017-10-24 16:51:33
//
//
//Function:
//Usage:
#include "digiter-decoder.h"

int main(int argc, char** argv){
    speakin::DigiterDecoderConfig config;
    speakin::DigiterDecoder decoder(config);

    std::string model_filename =argv[1],
                fst_filename = argv[2],
                wave_filename_scp = argv[3],
                word_syms_filename = argv[4];

    std::vector<int32> alignment, words;
    fst::SymbolTable *word_syms = NULL;
    word_syms = fst::SymbolTable::ReadText(word_syms_filename);
    kaldi::SequentialTokenReader wave_file_reader(wave_filename_scp);
    decoder.ReadModel(model_filename, fst_filename, word_syms_filename);
    
    for(; !wave_file_reader.Done(); wave_file_reader.Next()){
        //std::string wave_output_path = wave_file_reader.Value();
        std::string wave_filename = wave_file_reader.Key();
        //std::cout << "wave output path: " << wave_output_path << std::endl;
        std::cout << "(LOG)wave input path: " << wave_filename << std::endl;
    
        decoder.Decode(wave_filename, &alignment, &words);
        
        std::cout << "(LOG)RECOGNIZE RESULT: ";
        for( size_t i = 0; i < words.size(); i++ ) {
            std::string s = word_syms->Find(words[i]);
            std::cout << s << ' ';
        }
        std::cout << std::endl;
    }
}
