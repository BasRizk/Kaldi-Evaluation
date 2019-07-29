# -*- coding: utf-8 -*-

#!/usr/bin/env python

from __future__ import print_function

from kaldi.asr import NnetLatticeFasterOnlineRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleLoopedComputationOptions
from kaldi.online2 import (OnlineEndpointConfig,
                           OnlineIvectorExtractorAdaptationState,
                           OnlineNnetFeaturePipelineConfig,
                           OnlineNnetFeaturePipelineInfo,
                           OnlineNnetFeaturePipeline,
                           OnlineSilenceWeighting)
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader

from os import path, makedirs
from utils import cpu_info, gpu_info, prepare_pathes
from timeit import default_timer as timer
from jiwer import wer  
import platform, os, sys
import soundfile as sf
import pandas as pd  
import time


IS_RECURSIVE_DIRECTORIES = True
IS_TSV = False
USING_GPU = False
if USING_GPU:
    from kaldi import cudamatrix
    print("Using GPU support.")
    cudamatrix.CuDevice.instantiate().select_gpu_id("yes")
VERBOSE = True

#TEST_PATH = "tests/LibriSpeech_test-clean/test-clean"
TEST_PATH = "tests/LibriSpeech_test-other/test-other"
#TEST_PATH = "tests/iisys"
assert(path.exists(TEST_PATH))

try:
    TEST_CORPUS = TEST_PATH.split("/")[1]
except:
    print("WARNING: Path 2nd index does not exist.\n")

if  TEST_CORPUS == "iisys":
    IS_TSV = True
    IS_RECURSIVE_DIRECTORIES = False
else:
    IS_TSV = False
    IS_RECURSIVE_DIRECTORIES = True

try:
    if TEST_PATH.split("/")[2] == "Sprecher":
        AUDIO_INPUT="flac"
except:
    print("WARNING: Path 3rd index does not exist.\n")

# =============================================================================
# ------------------------Documenting Machine ID
# =============================================================================
localtime = time.strftime("%Y%m%d-%H%M%S")
platform_id = platform.machine() + "_" + platform.system() + "_" +\
                platform.node() + "_" + localtime
                
                
if USING_GPU:
    platform_id += "_use_gpu"

if TEST_CORPUS:
    platform_id = TEST_CORPUS + "_" + platform_id

platform_meta_path = "logs/online/" + platform_id

if not path.exists(platform_meta_path):
    makedirs(platform_meta_path)

if(USING_GPU):
    with open(os.path.join(platform_meta_path,"gpu_info.txt"), 'w') as f:
        f.write(gpu_info())
else:
    with open(os.path.join(platform_meta_path,"cpu_info.txt"), 'w') as f:
        f.write(cpu_info())

# =============================================================================
# ------------------------------Preparing pathes
# =============================================================================
model_dir = "exp/nnet3_chain/tdnn_f/"
conf_dir  = "conf"
ivectors_conf_dir = "conf"
model_path = path.join(model_dir, "final.mdl")
graph_path = path.join(model_dir, "graph/HCLG.fst")
symbols_path = path.join(model_dir, "graph/words.txt")
mfcc_hires_path = path.join(conf_dir, "mfcc_hires.conf")
online_config_path = path.join(conf_dir, "online_cmvn.conf")
if IS_RECURSIVE_DIRECTORIES:
    TEST_PATH_CUT = "/".join(TEST_PATH.split("/")[:-1])
else:
    TEST_PATH_CUT = TEST_PATH
scp_path = path.join(TEST_PATH_CUT, "wav.scp")
ivector_extractor_path = path.join(ivectors_conf_dir,"ivector_extractor.conf")
spk2utt_path = path.join(TEST_PATH_CUT,"spk2utt")
assert(path.exists(model_path))
assert(path.exists(graph_path))
assert(path.exists(symbols_path))
assert(path.exists(mfcc_hires_path))
assert(path.exists(online_config_path))
assert(path.exists(scp_path))
assert(path.exists(ivector_extractor_path))
assert(path.exists(spk2utt_path))

#localtime = time.strftime("%Y%m%d-%H%M%S")
log_filepath = platform_meta_path  +"/logs_" + localtime + ".txt"
summ_filepath = platform_meta_path  +"/summ_" + localtime + ".txt"
out_decode_path = path.join(platform_meta_path, "decode.out")
benchmark_filepath = platform_meta_path  +"/kaldi-asr_benchmark_ " + localtime + ".csv"
test_directories = prepare_pathes(TEST_PATH, recursive = IS_RECURSIVE_DIRECTORIES)
text_pathes = list()
text_file_exten = "txt"
if IS_TSV:
    text_file_exten = "tsv"
for d in test_directories:
    text_pathes.append(prepare_pathes(d, text_file_exten, recursive = False))
text_pathes.sort()  

# =============================================================================
# ----------------------------- Model Loading 
# =============================================================================
log_file = open(log_filepath, "w")
summ_file = open(summ_filepath, "w")

chunk_size = 1440

# Define online feature pipeline
#feats_args = "--mfcc-config="  + mfcc_hires_path + " " +\
#                    "--ivector-extraction-config=" + ivector_extractor_path +\
#                    "-verbose=1"

                
feat_opts = OnlineNnetFeaturePipelineConfig()
endpoint_opts = OnlineEndpointConfig()
po = ParseOptions("")
feat_opts.register(po)
endpoint_opts.register(po)
po.read_config_file(online_config_path)
feat_info = OnlineNnetFeaturePipelineInfo.from_config(feat_opts)


# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleLoopedComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150

print('Loading inference model from files\n {} \n {} \n {}\n'\
      .format(model_path, graph_path, symbols_path),
          file=sys.stderr)
log_file.write('Loading inference model from files\n {} \n {} \n {}\n'\
               .format(model_path, graph_path, symbols_path))
inf_model_load_start = timer()
asr = NnetLatticeFasterOnlineRecognizer.from_files(
    model_path, graph_path, symbols_path,
    decoder_opts=decoder_opts,
    decodable_opts=decodable_opts,
    endpoint_opts=endpoint_opts)
inf_model_load_end = timer() - inf_model_load_start
print('Loaded inference model in {:.3}s.\n'.format(inf_model_load_end),
      file=sys.stderr)
log_file.write('Loaded inference model in {:.3}s.\n'.format(inf_model_load_end))
summ_file.write('Loaded inference model in, {:.3}\n'.format(inf_model_load_end))
#
## Define feature pipelines as Kaldi rspecifiers
#feats_rspec = (
#    "ark:compute-mfcc-feats --config=" + mfcc_hires_path + " scp:" + scp_path +" ark:- |"
#                )
#ivectors_rspec = (
#    "ark:compute-mfcc-feats --config=" + mfcc_hires_path + " scp:"+ scp_path + " ark:- |"
#    "ivector-extract-online2 --config=" + ivector_extractor_path + " ark:" + spk2utt_path + " ark:- ark:- |"
#)
#
#log_file.write('feat_rspec \n{}\n'.format(feats_rspec))
#log_file.write('ivectors_rspec \n{}\n'.format(ivectors_rspec))


# =============================================================================
# ---Running the Kaldi STT Engine by running through the audio files
# =============================================================================
processed_data = "filename,length(sec),proc_time(sec),wer,actual_text,processed_text\n"
avg_wer = 0
avg_proc_time = 0
current_audio_number = 1

text_pathes = [path for subpathes in text_pathes for path in subpathes]

if IS_TSV:
    audio_transcripts = pd.concat( [ pd.read_csv(text_path, sep='\t', header=None, engine='python') for text_path in text_pathes] )
    audio_transcripts.sort_values(by = 0)
else:
    audio_transcripts = pd.concat( [ pd.read_csv(text_path, header=None, engine='python') for text_path in text_pathes] )
    audio_transcripts.sort_values(by = 0)
    audio_transcripts = audio_transcripts[0].str.split(" ", 1, expand = True)
audio_transcripts[1] = audio_transcripts[1].str.lower()
audio_transcripts = audio_transcripts.set_index(0)[1].to_dict()


# Decode (whole utterance)
num_of_audiofiles  = 0
for key, wav in SequentialWaveReader("scp:" + scp_path):
    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
    asr.set_input_pipeline(feat_pipeline)
    feat_pipeline.accept_waveform(wav.samp_freq, wav.data()[0])
    feat_pipeline.input_finished()
    
    audio_path = key
    try:
        audio, fs = sf.read(audio_path, dtype='int16')
    except:
        if VERBOSE: 
            print("# WARNING :: Audio File" + audio_path + " not readable.\n")
        log_file.write("# WARNING :: Audio File " + audio_path + " not readable.\n")
        continue
    audio_len = len(audio)/fs 
    print('Running inference.\n', file=sys.stderr)
    inference_start = timer()
    out = asr.decode()
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.\n' % (inference_end, audio_len))
    proc_time = inference_end
    proc_time = round(proc_time,3)
    
    # Processing WORD ERROR RATE (WER)
    processed_text = out['text'].lower()
    audio_filename = audio_path.split("/")[-1].split(".")[0]
    actual_text = audio_transcripts.get(audio_filename)
    if not actual_text:
        if VERBOSE: 
            print("# WARNING :: Transcript of file " + audio_filename + " does not exist.\n")
        log_file.write("# WARNING :: Transcript of file " + audio_filename + " does not exist.\n")
        continue
        
    num_of_audiofiles +=1
    current_wer = wer(actual_text, processed_text, standardize=True)
    current_wer = round(current_wer,3)
    
    # Accumlated data
    avg_proc_time += (proc_time/(audio_len))
    avg_wer += current_wer
    
    audio_path = audio_path.split("/")[-1]
    progress_row = audio_path + "," + str(audio_len) + "," + str(proc_time)  + "," +\
                    str(current_wer) + "," + actual_text + "," + processed_text
                     
    if(VERBOSE):
        print("# Audio number " + str(num_of_audiofiles) + "\n" +\
		  "# File (" + audio_path + "):\n" +\
              "# - " + str(audio_len) + " seconds long.\n"+\
              "# - actual    text: '" + actual_text + "'\n" +\
              "# - processed text: '" + processed_text + "'\n" +\
              "# - processed in "  + str(proc_time) + " seconds.\n"
              "# - WER = "  + str(current_wer) + "\n")
              
    log_file.write("# Audio number " + str(num_of_audiofiles) + "\n" +\
	      "# File (" + audio_path + "):\n" +\
          "# - " + str(audio_len) + " seconds long.\n"+\
          "# - actual    text: '" + actual_text + "'\n" +\
          "# - processed text: '" + processed_text + "'\n" +\
          "# - processed in "  + str(proc_time) + " seconds.\n"
          "# - WER = "  + str(current_wer) + "\n")
    
              
    processed_data+= progress_row + "\n"
    print(key, out["text"], flush=True)




#
## Decode (chunked + partial output)
#for key, wav in SequentialWaveReader("scp:wav.scp"):
#    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
#    asr.set_input_pipeline(feat_pipeline)
#    asr.init_decoding()
#    data = wav.data()[0]
#    last_chunk = False
#    part = 1
#    prev_num_frames_decoded = 0
#    for i in range(0, len(data), chunk_size):
#        if i + chunk_size >= len(data):
#            last_chunk = True
#        feat_pipeline.accept_waveform(wav.samp_freq, data[i:i + chunk_size])
#        if last_chunk:
#            feat_pipeline.input_finished()
#        asr.advance_decoding()
#        num_frames_decoded = asr.decoder.num_frames_decoded()
#        if not last_chunk:
#            if num_frames_decoded > prev_num_frames_decoded:
#                prev_num_frames_decoded = num_frames_decoded
#                out = asr.get_partial_output()
#                print(key + "-part%d" % part, out["text"], flush=True)
#                part += 1
#    asr.finalize_decoding()
#    out = asr.get_output()
#    print(key + "-final", out["text"], flush=True)
#
## Decode (chunked + partial output + endpointing
##         + ivector adaptation + silence weighting)
#adaptation_state = OnlineIvectorExtractorAdaptationState.from_info(
#    feat_info.ivector_extractor_info)
#for key, wav in SequentialWaveReader("scp:wav.scp"):
#    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
#    feat_pipeline.set_adaptation_state(adaptation_state)
#    asr.set_input_pipeline(feat_pipeline)
#    asr.init_decoding()
#    sil_weighting = OnlineSilenceWeighting(
#        asr.transition_model, feat_info.silence_weighting_config,
#        decodable_opts.frame_subsampling_factor)
#    data = wav.data()[0]
#    last_chunk = False
#    utt, part = 1, 1
#    prev_num_frames_decoded, offset = 0, 0
#    for i in range(0, len(data), chunk_size):
#        if i + chunk_size >= len(data):
#            last_chunk = True
#        feat_pipeline.accept_waveform(wav.samp_freq, data[i:i + chunk_size])
#        if last_chunk:
#            feat_pipeline.input_finished()
#        if sil_weighting.active():
#            sil_weighting.compute_current_traceback(asr.decoder)
#            feat_pipeline.ivector_feature().update_frame_weights(
#                sil_weighting.get_delta_weights(
#                    feat_pipeline.num_frames_ready()))
#        asr.advance_decoding()
#        num_frames_decoded = asr.decoder.num_frames_decoded()
#        if not last_chunk:
#            if asr.endpoint_detected():
#                asr.finalize_decoding()
#                out = asr.get_output()
#                print(key + "-utt%d-final" % utt, out["text"], flush=True)
#                offset += int(num_frames_decoded
#                              * decodable_opts.frame_subsampling_factor
#                              * feat_pipeline.frame_shift_in_seconds()
#                              * wav.samp_freq)
#                feat_pipeline.get_adaptation_state(adaptation_state)
#                feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
#                feat_pipeline.set_adaptation_state(adaptation_state)
#                asr.set_input_pipeline(feat_pipeline)
#                asr.init_decoding()
#                sil_weighting = OnlineSilenceWeighting(
#                    asr.transition_model, feat_info.silence_weighting_config,
#                    decodable_opts.frame_subsampling_factor)
#                remainder = data[offset:i + chunk_size]
#                feat_pipeline.accept_waveform(wav.samp_freq, remainder)
#                utt += 1
#                part = 1
#                prev_num_frames_decoded = 0
#            elif num_frames_decoded > prev_num_frames_decoded:
#                prev_num_frames_decoded = num_frames_decoded
#                out = asr.get_partial_output()
#                print(key + "-utt%d-part%d" % (utt, part),
#                      out["text"], flush=True)
#                part += 1
#    asr.finalize_decoding()
#    out = asr.get_output()
#    print(key + "-utt%d-final" % utt, out["text"], flush=True)
#    feat_pipeline.get_adaptation_state(adaptation_state)


# =============================================================================
# ---------------Finalizing processed data and Saving Logs
# =============================================================================
        
avg_proc_time /= num_of_audiofiles
avg_wer /= num_of_audiofiles
if(VERBOSE):
    print("Avg. Proc. time (sec/second of audio) = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
log_file.write("Avg. Proc. time/sec = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
summ_file.write("Avg. Proc. time/sec," + str(avg_proc_time) + "\n" +\
          "Avg. WER," + str(avg_wer))
log_file.close()
summ_file.close()
processed_data+= "AvgProcTime (sec/second of audio)," + str(avg_proc_time) + "\n"
processed_data+= "AvgWER," + str(avg_wer) + "\n"


with open(benchmark_filepath, 'w') as f:
    for line in processed_data:
        f.write(line)
