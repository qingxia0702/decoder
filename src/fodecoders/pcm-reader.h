//home/huanghe/workspace/keyword-decoder/src/pcm-reader.h
//
//Author: Pro.Dogegg(Heqing Huang)
//Time:   2017-10-26 12:51:33
//
//
//Function:
//Usage:
#ifndef _PCM_READER_H
#define _PCM_READER_H
#endif
#include "base/kaldi-common.h"
#include "base/kaldi-types.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"

namespace fotalk{
struct PcmReaderConfig{
    int32 samp_freq;
    int32 channels;
    bool reverse_bytes_;

    PcmReaderConfig():
            samp_freq(16000),
            channels(1),
            reverse_bytes_(false){};
};

class PcmReader {
    kaldi::Matrix<kaldi::BaseFloat> data_;
    PcmReaderConfig config_;

 public:
    PcmReader(PcmReaderConfig config){
        config_ = config;
    }
    void Read(std::istream &is){
        const uint32 kBlockSize = 1024*1024;
        std::vector<char> buffer;
        data_.Resize(0,0);
        
        while(is){
           uint32  offset = buffer.size();
           buffer.resize(offset + kBlockSize);
           is.read(&buffer[offset], kBlockSize);
           uint32 bytes_read = is.gcount();
           buffer.resize(offset + bytes_read);
        }
        
        uint16 *data_ptr = reinterpret_cast<uint16*>(&buffer[0]);
        data_.Resize(config_.channels, buffer.size()/2*config_.channels);
        for(int32 i = 0; i < data_.NumCols(); i++){
            for(int32 j = 0; j< data_.NumRows(); j++){
                int16 k = *data_ptr++;
                //if(config_.reverse_bytes_)
                    //KAILDI_SWAP2(k);
                data_(j,i) = k;
            }
        }
    }
    const kaldi::Matrix<kaldi::BaseFloat>& Data() const{
       return data_; 
    }
};
}//namespace fotalk

