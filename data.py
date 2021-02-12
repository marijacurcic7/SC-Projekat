import pandas as pd

import librosa
import librosa.display

import numpy as np



def load_data():
    folder_path = "dataset/"
    folder_list = ["Accordion", "Clarinet_Bb", "Contrabass", "Horn", "Viola", "Violin", "Violoncello"]
    features = []
    onlyfiles = []

    from os import listdir
    from os.path import isfile, join

    for folder_inst in folder_list:
        onlyfiles = onlyfiles + [f for f in listdir("dataset/" + folder_inst) if
                                 isfile(join("dataset/" + folder_inst + "/", f))]

    for file in onlyfiles:
        dodatak = ''
        if (file.split("-")[0] == "Acc"):
            dodatak = "Accordion"
        elif (file.split("-")[0] == "ClBb"):
            dodatak = "Clarinet_Bb"
        elif (file.split("-")[0] == "Cb"):
            dodatak = "Contrabass"
        elif (file.split("-")[0] == "Hn"):
            dodatak = "Horn"
        elif (file.split("-")[0] == "Va"):
            dodatak = "Viola"
        elif (file.split("-")[0] == "Vn"):
            dodatak = "Violin"
        elif (file.split("-")[0] == "Vc"):
            dodatak = "Violoncello"

        file_name = folder_path + dodatak + "/" + file
        a = file.split("-")
        instrument = a[0]
        # print(file_name)
        pitch = a[2]
        if (len(pitch) == 3):
            pitch = pitch[:2]
        else:
            pitch = pitch[:1]
        data = extract_spectrogram(file_name)
        features.append([file_name, data, instrument, pitch])

    # menjanje boje slike
    features_gray = []
    for list in features:
        imggray = my_rgb2gray(list[1])
        features_gray.append([list[0], normalize_gray(imggray), list[2], list[3]])

    return features_gray


def extract_spectrogram(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast", sr=22050)
        a, index = librosa.effects.trim(audio, top_db=30, frame_length=2048, hop_length=512)
        y_out = a[:44100]
        spectrogram = librosa.feature.melspectrogram(y=y_out, sr=sample_rate, n_fft=2048, hop_length=1024)
        spec_shape = spectrogram.shape
        if (spec_shape[1] < 44):
            print(spec_shape)
            print(file_name)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram

    except Exception as e:
        print(e)
        print("Error encountered while parsing file: ", file_name)
        return None

def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)
    img_gray = 0.21*img_rgb[:, :] + 0.72*img_rgb[:, :]
    return img_gray

def normalize_gray(array):
    return (array - array.min())/(array.max() - array.min())