# -*- coding: utf-8 -*-

from os import listdir, path, makedirs
from os.path import splitext
#from pydub import AudioSegment
import soundfile as sf

def prepare_pathes(directory, exten = '', global_dir=False):
    updated_pathes = list()
    if(global_dir):
        subdirectories = listdir(directory)
        subdirectories.sort()
        for subdirectory in subdirectories:
            subdirectory = path.join(directory, subdirectory)
            filenames = listdir(subdirectory)
            filenames.sort()
            for filename in filenames:
                if(filename.endswith(exten)):
                    updated_pathes.append(path.join(subdirectory, filename))
            updated_pathes.sort()
    else:
        filenames = listdir(directory)
        for filename in filenames:
            if(filename.endswith(exten)):
                updated_pathes.append(path.join(directory, filename))
    updated_pathes.sort()
    return updated_pathes

def wav2flac(flac_path, save_dir = None):
    wav_path = "%s.wav" % splitext(flac_path)[0]
    audio, rate = sf.read(flac_path)
    if save_dir:
        wav_path = path.join(save_dir, wav_path)  
        dir_path = "/".join(wav_path.split("/")[:-1])
        if not path.exists(dir_path):
            makedirs(dir_path)
    sf.write(wav_path, audio, rate, format = "WAV")
    
if __name__ == "__main__":
    import sys
try:
    flac_pathes = prepare_pathes(sys.argv[1])
except:
    print("Not a directory.\n")
    flac_pathes = []
    for i in range(1, len(sys.argv)):
        flac_pathes.append(sys.argv[i])
save_dir = None
#if sys.argv[2]:
#    save_dir = sys.argv[2]

num_of_files_converted = 0
for flac_path in flac_pathes:
#    print(flac_path + "\n")
    wav2flac(flac_path, save_dir)
    num_of_files_converted += 1
    
print("Num of files converted = %d", num_of_files_converted)