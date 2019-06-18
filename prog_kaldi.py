# -*- coding: utf-8 -*-
"""
Testing performance of Mozilla Deepspeech on different devices

Created on Tue Mar 21 2019

@author: ironbas3
"""
from __future__ import print_function

from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader, CompactLatticeWriter
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

platform_meta_path = "logs/" + platform_id

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
assert(path.exists(scp_path))
assert(path.exists(ivector_extractor_path))
assert(path.exists(spk2utt_path))

#localtime = time.strftime("%Y%m%d-%H%M%S")
log_filepath = platform_meta_path  +"/logs_" + localtime + ".txt"
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
# Instantiate the recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150

print('Loading inference model from files\n {} \n {} \n {}\n'\
      .format(model_path, graph_path, symbols_path),
          file=sys.stderr)
log_file.write('Loading inference model from files\n {} \n {} \n {}\n'\
               .format(model_path, graph_path, symbols_path))
inf_model_load_start = timer()
asr = NnetLatticeFasterRecognizer.from_files(
    model_path, graph_path, symbols_path,
    decoder_opts=decoder_opts,
    decodable_opts=decodable_opts)
inf_model_load_end = timer() - inf_model_load_start
print('Loaded inference model in {:.3}s.\n'.format(inf_model_load_end),
      file=sys.stderr)
log_file.write('Loaded inference model in {:.3}s.\n'.format(inf_model_load_end))

# Define feature pipelines as Kaldi rspecifiers
feats_rspec = (
    "ark:compute-mfcc-feats --config=" + mfcc_hires_path + " scp:" + scp_path +" ark:- |"
                )
ivectors_rspec = (
    "ark:compute-mfcc-feats --config=" + mfcc_hires_path + " scp:"+ scp_path + " ark:- |"
    "ivector-extract-online2 --config=" + ivector_extractor_path + " ark:" + spk2utt_path + " ark:- ark:- |"
)

log_file.write('feat_rspec \n{}\n'.format(feats_rspec))
log_file.write('ivectors_rspec \n{}\n'.format(ivectors_rspec))

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


# =============================================================================
# ---Running the Kaldi STT Engine by running through the audio files
# =============================================================================


num_of_audiofiles  = 0
with SequentialMatrixReader(feats_rspec) as f, \
     SequentialMatrixReader(ivectors_rspec) as i, \
     open(out_decode_path, "w") as o:
    for (key, feats), (_, ivectors) in zip(f, i):
        
        audio_path = key
        try:
            audio, fs = sf.read(audio_path, dtype='int16')
        except:
            if VERBOSE: 
                print("# WARNING :: Audio File" + audio_path + " not readable.\n")
            log_file.write("# WARNING :: Audio File " + audio_path + " not readable.\n")
            continue
        audio_len = len(audio)/fs 
        n_input = 2
        print('Running inference.\n', file=sys.stderr)
        inference_start = timer()
        out = asr.decode((feats, ivectors))
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
            print("# File (" + audio_path + "):\n" +\
                  "# - " + str(audio_len) + " seconds long.\n"+\
                  "# - actual    text: '" + actual_text + "'\n" +\
                  "# - processed text: '" + processed_text + "'\n" +\
                  "# - processed in "  + str(proc_time) + " seconds.\n"
                  "# - WER = "  + str(current_wer) + "\n")
                  
        log_file.write("# File (" + audio_path + "):\n" +\
              "# - " + str(audio_len) + " seconds long.\n"+\
              "# - actual    text: '" + actual_text + "'\n" +\
              "# - processed text: '" + processed_text + "'\n" +\
              "# - processed in "  + str(proc_time) + " seconds.\n"
              "# - WER = "  + str(current_wer) + "\n")
        
                  
        processed_data+= progress_row + "\n"
        
        print(key, out["text"], file=o)

#lat_dir = path.join(data_dir, "lat.gz")
#lat_wspec = "ark:| gzip -c > " + lat_dir

## Extract the features, decode and write output lattices
#with SequentialMatrixReader(feats_rspec) as feats_reader, \
#     SequentialMatrixReader(ivectors_rspec) as ivectors_reader, \
#     CompactLatticeWriter(lat_wspec) as lat_writer:
#    for (fkey, feats), (ikey, ivectors) in zip(feats_reader, ivectors_reader):
#        assert(fkey == ikey)
#        out = asr.decode((feats, ivectors))
#        print(fkey, out["text"])
#        lat_writer[fkey] = out["lattice"]        


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
log_file.close()
processed_data+= "AvgProcTime (sec/second of audio)," + str(avg_proc_time) + "\n"
processed_data+= "AvgWER," + str(avg_wer) + "\n"


with open(benchmark_filepath, 'w') as f:
    for line in processed_data:
        f.write(line)
    
        
        
