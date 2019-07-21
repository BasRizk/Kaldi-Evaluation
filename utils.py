# -*- coding: utf-8 -*-

from os import listdir, path
import platform, os

##############################################################################
# ------------------------Documenting Machine ID
##############################################################################

def cpu_info():
    if platform.system() == 'Windows':
        return platform.processor()
    elif platform.system() == 'Darwin':
        command = '/usr/sbin/sysctl -n machdep.cpu.brand_string'
        return os.popen(command).read().strip()
    elif platform.system() == 'Linux':
        command = 'cat /proc/cpuinfo'
        return os.popen(command).read().strip()
    return 'platform not identified'

def gpu_info():
    if platform.system() == 'Linux':
        command = 'nvidia-smi'
        if not os.popen(command).read().strip() == "":
            return os.popen(command).read().strip()
        return "Probably running on Jetson NANO with Nvidia MaxWel GPU - Tegra"
    return 'platform not identified'


#def machine_info(USING_GPU = False):
#    localtime = time.strftime("%Y%m%d-%H%M%S")
#    platform_id = platform.machine() + "_" + platform.system() + "_" +\
#                    platform.node() + "_" + localtime
#    platform_meta_path = "logs/" + platform_id
#    
#    if not path.exists(platform_meta_path):
#        makedirs(platform_meta_path)
#    
#    if(USING_GPU):
#        with open(os.path.join(platform_meta_path,"gpu_info.txt"), 'w') as f:
#            f.write(gpu_info())
#    else:
#        with open(os.path.join(platform_meta_path,"cpu_info.txt"), 'w') as f:
#            f.write(cpu_info())
#    
#    return localtime, platform_id, platform_meta_path

##############################################################################
# ------------------------------Preparing pathes
##############################################################################

def prepare_pathes(directory, exten = '', recursive=True):
    if not path.isdir(directory):
        print(directory + " is not a directory.")
        return
    updated_pathes = list()
    if recursive:
        subdirectories = listdir(directory)
        subdirectories.sort()
        for subdirectory in subdirectories:
            subdirectory = path.join(directory, subdirectory)
            if not path.isdir(subdirectory):
                print(subdirectory + " is not a directory.")
                continue
            filenames = listdir(subdirectory)
            filenames.sort()
            for filename in filenames:
                if(filename.endswith(exten)):
                    updated_pathes.append(path.join(subdirectory, filename))
            updated_pathes.sort()
    else:
        # filenames or sub directories
        subdirectories = listdir(directory)
        for subdir in subdirectories:
            subdir = path.join(directory, subdir)
            if (path.isdir(subdir) and exten == ''):
                updated_pathes.append(subdir)
            elif(subdir.endswith(exten) and not exten == ''):
                updated_pathes.append(subdir)
                
    updated_pathes.sort()
    return updated_pathes

