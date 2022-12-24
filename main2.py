from flask import Flask, render_template, send_file, request, redirect, jsonify
import os

import pickle
import librosa
import numpy as np
import sounddevice as sd
import wavio as wv
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy.io.wavfile as wavfile
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import librosa.display
import pandas as pd
import SpeakerIdentification


app = Flask(__name__)


class variables:
    counter = 0


def features_spectogram(feature_name, feature):

    # We'll show each in its own subplot
    fig = plt.figure(figsize=(6, 6))
    # fig.patch.set_facecolor('black')
    librosa.display.specshow(feature)
    plt.ylabel(feature_name)
    plt.colorbar()

    image_file_name = 'static/assets/images/'+feature_name + '.jpg'
    plt.savefig(image_file_name)


def draw_mel(sr, mel_Spectrogram, fet_name):
    fig = plt.figure(figsize=(6, 6))
    S_dB = librosa.power_to_db(mel_Spectrogram, ref=np.max)
    # fig.patch.set_facecolor('black')
    librosa.display.specshow(S_dB)
    plt.colorbar()
    image_file_name = 'static/assets/images/'+fet_name + '.jpg'
    plt.savefig(image_file_name)


def draw_contrast(Spectrogram, sr, fet_name):
    fig = plt.figure(figsize=(6, 6))
    contrast = librosa.feature.spectral_contrast(S=Spectrogram, sr=sr)
    # fig.patch.set_facecolor('black')
    librosa.display.specshow(contrast)
    plt.colorbar()
    image_file_name = 'static/assets/images/'+fet_name + '.jpg'
    plt.savefig(image_file_name)


def draw_centroid(Spectrogram, fet_name):

    cent = librosa.feature.spectral_centroid(S=Spectrogram)
    times = librosa.times_like(cent)
    fig, ax = plt.subplots()
    # ax.tick_params(colors='white',which= 'both')
    # fig.patch.set_facecolor('black')
    centroid_img = librosa.display.specshow(librosa.amplitude_to_db(Spectrogram, ref=np.max),
                                            y_axis='log', x_axis='time', ax=ax)
    fig.colorbar(centroid_img)
    ax.plot(times, cent.T, label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    # ax.set(title='log Power spectrogram')
    image_file_name = 'static/assets/images/'+fet_name + '.jpg'
    plt.savefig(image_file_name)


def draw(file_name):
    file_name = 'testing_set\sample.wav'
    signal, sr = librosa.load(file_name)
    Spectrogram = np.abs(librosa.stft(signal))
    mel_Spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128,
                                                     fmax=8000)

    draw_mel(sr, mel_Spectrogram, 'mel')
    draw_contrast(Spectrogram, sr, 'contrast')
    draw_centroid(Spectrogram, 'centroid')


def prepare_testing(to_test):

    features = []
    # reading records
    y, sr = librosa.load(to_test)
    # remove leading and trailing silence
    y, index = librosa.effects.trim(y)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    features_spectogram('mfcc', mfcc)

    # features_spectogram ('rms',rmse)

    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'

    features.append(to_append.split())

    for counter in range(0, len(features[0])):
        features[0][counter] = float(features[0][counter])
    # print (features)
    return features


def test_model(wav_file):
    # wav_file='k_close_2.wav'
    # wav_file='a_open_8.wav'

    wav_file = '(4).wav'
    features = prepare_testing(wav_file)
    model = pickle.load(open('model_random2.pkl', 'rb'))
    model_output = model.predict(features)
    print(model_output[0])

    if model_output[0] == 0:
        result = 'Close the door'
    elif model_output[0] == 1:
        result = 'Open the door'
    else:
        result = 'not a correct sentence'

    # print('reeeeeeesult---------------------')
    # print(result)
    return result


def predict_sound(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)

    features = []

    features.append(np.concatenate(
        (mfccs, chroma, mel, contrast, tonnetz), axis=0))
    open_model = pickle.load(open("audio_identefire__new__one.pkl", 'rb'))
    result = open_model.predict(features)
    print(result)
    return result


def visualize(y1, y2, score):
    # fig, ax = plt.subplots(2, figsize=(8, 8))
    # print(y1, y2)
    # x1 = ("Abram", "Hager", "Mariem", "Naira")
    # x2 = ("open the door", "wrong sentence", "open the gate", 'close ')
    # ax[0].bar(x1, y1, align='center')
    # ax[1].bar(x2, y2, align='center')
    # ax[0].set_xlabel('speaker')
    # ax[0].set_ylabel('score')
    # ax[1].set_xlabel('sentence')
    # ax[1].set_ylabel('score')
    # fig.patch.set_facecolor('black')
    # ax[0].tick_params(colors='white', which='both')
    # ax[1].tick_params(colors='white', which='both')
    # # for i in range(len(y1)):
    # #     ax[0].hlines(y1[i], 0, x1[i])
    # #     ax[1].hlines(y2[i], 0, x2[i])

    fig, ax = plt.subplots(1, figsize=(10, 8))
    x1 = ("A OD ", "A CD ", "A OG ", "A CG  ",
          "H OD ", "H CD ", "H OG ", "H CG ",
          "M OD ", "M CD ", "M OG ", "M CG",
          "N OD ", "N CD ", "N OG ", "N CG ")
    ax.bar(x1, abs(score), align='center')
    ax.set_xlabel('speakers')
    ax.set_ylabel('score')
    # fig.patch.set_facecolor('black')
    # ax.tick_params(colors='white', which='both')

    df = pd.DataFrame({'category': x1,
                       'number': abs(score)})
    df.set_index('category', inplace=True)
    df.sort_values('number', inplace=True)

    ax = df.plot(y='number', kind='bar', legend=False)
    # plt.patch.set_facecolor('black')

    plt.savefig('static/assets/images/hesto.jpg')


@app.route("/", methods=["GET", "POST"])
def index():

    speech = ''
    speaker = ''
    file_name = ''
    img = ''

    y = []
    sr = []

    if request.method == "POST":

        speaker, speech, speaker_scores, sentence_scores, scores = SpeakerIdentification.start_testing()
        file_name = 'testing_set\sample.wav'
        draw(file_name)
        img = visualize(speaker_scores, sentence_scores, scores)
        img = 'static/assets/images/result'+str(variables.counter)+'.jpg'

    # return send_file(speech=speech,speaker=speaker,file_name=file_name,y=y,sr=sr)
    return render_template('index.html', speech=speech, speaker=speaker, file_name=file_name, y=y, sr=sr, img=img)

# @app.route("/", methods=["GET", "POST"])
# def index():

#         speech =''
#         speaker =''
#         file_name=''
#         img=''

#         y=[]
#         sr=[]


#         if request.method == "POST":

#             file_name='testing_set\sample.wav'

#             SpeakerIdentification.record_audio_test()
#             speaker_model1,speech=SpeakerIdentification.start_testing()
#             speaker_model2=predict_sound(file_name)
#             speaker=speaker_model1
#             # y, sr = librosa.load(file_name)
#             draw(file_name)
#             img= visualize(file_name)
#             img='static/assets/images/result'+str(variables.counter)+'.jpg'

#         # return send_file(speech=speech,speaker=speaker,file_name=file_name,y=y,sr=sr)
#         return render_template('index.html', speech=speech,speaker=speaker,file_name=file_name,y=y,sr=sr,img=img)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
