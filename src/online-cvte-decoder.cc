//cvte-decoder/src/online-cvte-decoder.cc
//
//Author:   Pro.Dogegg(Heqing Huang)
//Time:     2017-9-28 09:15:35
//Copyright 2017 SpeakIn
//
//
//Function: Decoding an input file.wav with cvte's acoustics model.
//Usage: online-cvte-decoder final.mdl HCLG.fst file.wav words.txt

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"


int main(int argc, char *argv[]) {
    try{
        //using namespace kaldi;
        //using namespace fst;
        typedef kaldi::int32 int32;
        typedef kaldi::int64 int64;
        
        kaldi::BaseFloat chunk_length_secs = 0.18;
        //Definition of options configuire classes
        kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
        decodable_opts.frame_subsampling_factor = 1;
        decodable_opts.frames_per_chunk = 50;
        decodable_opts.acoustic_scale = 1.0;
        
        //Init decoder configures
        kaldi::LatticeFasterDecoderConfig decoder_opts;
        decoder_opts.max_active = 7000;
        decoder_opts.beam = 15.0;
        decoder_opts.beam_delta = 8.0;

        //Init featrue congfiures
        kaldi::OnlineNnet2FeaturePipelineConfig feature_opts;
        feature_opts.feature_type = "fbank";
        feature_opts.fbank_config = "../configs/fbank.conf";

        std::string nnet3_rxfilename = argv[1],         //final.mdl
                    fst_rxfilename = argv[2],           //HCLG.fst
                    wav_rxfilename = argv[3],           //file.wav
                    word_syms_rxfilename = argv[4];     //words.txt

        kaldi::OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);
        kaldi::OnlineIvectorExtractorAdaptationState adaptation_state(
                feature_info.ivector_extractor_info);
        kaldi::OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);

        //Read final.mdl
        kaldi::TransitionModel trans_model;
        kaldi::nnet3::AmNnetSimple am_nnet;
        {
            bool binary;
            kaldi::Input ki(nnet3_rxfilename, &binary);
            trans_model.Read(ki.Stream(), binary);
            am_nnet.Read(ki.Stream(), binary);
            SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
            SetDropoutTestMode(true, &(am_nnet.GetNnet()));
            kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(),
                    &(am_nnet.GetNnet()));
        }

        //Read HCLG.fst
        fst::Fst<fst::StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_rxfilename);
        
        //Read file.wav
        kaldi::Input wave_input(wav_rxfilename);
        kaldi::WaveData wave_data;
        wave_data.Read(wave_input.Stream());
        kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);
        
        //Read words.txt
        fst::SymbolTable *word_syms = NULL;
        word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename);

        //Silence operations
        kaldi::OnlineSilenceWeighting silence_weighting(
                trans_model,
                feature_info.silence_weighting_config,
                decodable_opts.frame_subsampling_factor);
        
        //Definition decoder as SingleUtteranceNnet3Decoder
        kaldi::nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts, &am_nnet);
        kaldi::SingleUtteranceNnet3Decoder decoder(
                decoder_opts,
                trans_model,
                decodable_info,
                *decode_fst,
                &feature_pipeline);

        //Diagnostics of file.wav
        kaldi::BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs) {
            chunk_length = int32(samp_freq * chunk_length_secs);
            if (chunk_length == 0) chunk_length = 1;
        } else {
            chunk_length = std::numeric_limits<int32>::max();
        }
        
        //Do decoding for file.wav
        int32 samp_offset = 0;
        std::vector<std::pair<int32, kaldi::BaseFloat> > delta_weights;
        while (samp_offset < data.Dim()) {
            int32 samp_remaining = data.Dim() - samp_offset;
            int32 num_samp = chunk_length < samp_remaining ? chunk_length: samp_remaining;
            kaldi::SubVector<kaldi::BaseFloat> wave_part(data, samp_offset, num_samp);
            feature_pipeline.AcceptWaveform(samp_freq, wave_part);
            samp_offset += num_samp;

            if (samp_offset == data.Dim()) {
                feature_pipeline.InputFinished();
            }

            if (silence_weighting.Active() &&
                    feature_pipeline.IvectorFeature() != NULL) {
                silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
                silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                                 &delta_weights);
                feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
            }
            decoder.AdvanceDecoding();
        }//while end of the wave frame
        std::cout << "DEBUG: Begin the final decoding!" << std::endl;
        decoder.FinalizeDecoding();

        //Get the result of decoder
        kaldi::CompactLattice clat, best_path_clat;
        kaldi::Lattice best_path_lat;
        kaldi::LatticeWeight weight;
        std::vector<int32> alignment, words;
        bool end_of_wave = true;

        decoder.GetLattice(end_of_wave, &clat);
        kaldi::CompactLatticeShortestPath(clat, &best_path_clat);
        fst::ConvertLattice(best_path_clat, &best_path_lat);
        fst::GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
        std::cout << "DEBUG: The result size is:" << words.size() <<std::endl;

        for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "")
                KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
            std::cout<< s << ' ';
        }
        std::cerr << std::endl;
        
    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
}// main()

