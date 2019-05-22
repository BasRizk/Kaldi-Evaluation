# -*- coding: utf-8 -*-

from __future__ import print_function

from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader, CompactLatticeWriter
from os import listdir, path, makedirs
from jiwer import wer    
import time
import platform, os, sys
from utils import cpu_info, gpu_info, prepare_pathes
from timeit import default_timer as timer
import soundfile as sf


IS_GLOBAL_DIRECTORIES = True
USING_GPU = False
VERBOSE = True
# =============================================================================
# ------------------------Documenting Machine ID
# =============================================================================
localtime = time.strftime("%Y%m%d-%H%M%S")
platform_id = platform.machine() + "_" + platform.system() + "_" +\
                platform.node() + "_" + localtime
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
data_dir = "./librispeech-test"
model_dir = "exp/nnet3_chain/tdnn_f/"
conf_dir  = "conf"
ivectors_conf_dir = "conf"
model_path = path.join(model_dir, "final.mdl")
graph_path = path.join(model_dir, "graph/HCLG.fst")
symbols_path = path.join(model_dir, "graph/words.txt")
mfcc_hires_path = path.join(conf_dir, "mfcc_hires.conf")
scp_path = path.join(data_dir, "wav.scp")
ivector_extractor_path = path.join(ivectors_conf_dir,"ivector_extractor.conf")
spk2utt_path = path.join(data_dir, "spk2utt")
assert(path.exists(model_path))
assert(path.exists(graph_path))
assert(path.exists(symbols_path))
assert(path.exists(mfcc_hires_path))
assert(path.exists(scp_path))
assert(path.exists(ivector_extractor_path))
assert(path.exists(spk2utt_path))

localtime = time.strftime("%Y%m%d-%H%M%S")
log_filepath = platform_meta_path  +"/logs_" + localtime + ".txt"
out_decode_path = path.join(platform_meta_path, "decode.out")
benchmark_filepath = platform_meta_path  +"/kaldi-asr_benchmark_ " + localtime + ".csv"
test_directories = prepare_pathes(data_dir, global_dir=IS_GLOBAL_DIRECTORIES)
text_pathes = list()
for d in test_directories:
    text_pathes.append(prepare_pathes(d, "txt"))
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

with SequentialMatrixReader(feats_rspec) as f, \
     SequentialMatrixReader(ivectors_rspec) as i, \
     open(out_decode_path, "w") as o:
    for (key, feats), (_, ivectors) in zip(f, i):
        print('Running inference.', file=sys.stderr)
        print("\n\n\n KEY =\n" + key)
        inference_start = timer()
        out = asr.decode((feats, ivectors))
        inference_end = timer() - inference_start
#        print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_len))
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
# ---Evaluating results
# =============================================================================

import pandas as pd
decoded_output = pd.read_csv("decode.out", header=None)
decoded_output.sort_values(by = 0)
decoded_output = decoded_output[0].str.split(" ", 1, expand = True)
decoded_wav_filenames = decoded_output[0]
decoded_texts = decoded_output[1]

num_of_audiofiles = decoded_wav_filenames.size
processed_data = "filename,length(sec),proc_time(sec),wer,actual_text,processed_text\n"
avg_wer = 0
avg_proc_time = 0
current_audio_number = 1
for audio_group, audio_text_group_path in zip(audio_pathes, text_pathes):
    audio_transcripts = open(audio_text_group_path[0], 'r').readlines()
    audio_transcripts.sort()
    for audio_path, audio_transcript in zip(audio_group, audio_transcripts):
        
        print("\n=> Progress = " + "{0:.2f}".format((current_audio_number/num_of_audiofiles)*100) + "%\n" )
        current_audio_number+=1
        
        audio, fs = sf.read(audio_path, dtype='int16')
        audio_len = len(audio)/fs 

        #start_proc = time.time()
        print('Running inference.', file=sys.stderr)
        inference_start = timer()
        processed_text = ds.stt(audio, fs)
        inference_end = timer() - inference_start
        print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_len))
        #end = time.time()
       
        proc_time = inference_end
        proc_time = round(proc_time,3)

    
        # Processing WORD ERROR RATE (WER)
        audio_transcript = audio_transcript[:-1].split(" ")
        actual_text = " ".join(audio_transcript[1:]).lower()
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
