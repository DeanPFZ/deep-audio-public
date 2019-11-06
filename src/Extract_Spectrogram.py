## Import Dependencies ##
import numpy as np
import librosa as lib
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display as disp

## Define Directories ##
base_dir = r'G:\Audio Fingerprinting\Dataset'
esc_dir = r'G:\Audio Fingerprinting\Dataset\ESC-50-master\ESC-50-master'
meta_data = os.path.join(esc_dir ,'meta')
audio_dir = os.path.join(esc_dir,'audio')
spectrogram_dir = os.path.join(base_dir,'Spectrograms')


## Read CSV file and operate on spectrogram ##
d = pd.read_csv(os.path.join(meta_data,'esc50.csv'))
classes = np.asarray(d['target'])
file_list = d['filename']

## First create directories for specific classes ##


print ("Starting to work on the audio")

for idx,file in enumerate(file_list):
    audio = os.path.join(audio_dir,file)
    signal, sr  = lib.load(audio, sr = 44100, mono = True)
    spec = lib.feature.melspectrogram(signal, sr=44100, n_fft=2205, hop_length=441)
    # fig = dp.figure()
    f, ax = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(8, 2))
    ax.imshow(spec, origin='lower', interpolation=None, cmap='viridis', aspect=1.1)
    image_name = os.path.join(spectrogram_dir,str(file[:-4] +'.png'))
    f.tight_layout()
    # str(classes[idx])
    # fig.savefig(image_name)
    print(image_name)
    #plt.show()
    plt.savefig(image_name, bbox_inches='tight', dpi=72)
    plt.close()

print("Spectrogram generation is done!!!")
