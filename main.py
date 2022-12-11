from flask import Flask, render_template, send_file, request, redirect,jsonify
import os
import speech_recognition as sr
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
import wave
from PIL import Image 
import PIL 



app = Flask(__name__)
class variables:
    counter=0

# @app.route("/receive", methods=['post'])
# def form():
#     files = request.files
#     file = files.get('file')
#     print(file)

#     with open(os.path.abspath(f'backend/audios/{file}'), 'wb') as f:
#         f.write(file.content)

#     response = jsonify("File received and saved!")
#     response.headers.add('Access-Control-Allow-Origin', '*')

#     return response

def prepare_testing(to_test):

    features=[]
    # reading records
    y, sr       = librosa.load(to_test)
    # remove leading and trailing silence
    y, index    = librosa.effects.trim(y)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse        = librosa.feature.rms(y=y)
    spec_cent   = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw     = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(y)
    mfcc        = librosa.feature.mfcc(y=y, sr=sr)

    to_append   = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    
    features.append(to_append.split())

    
    for counter in range(0,len(features[0])):
        features[0][counter]=float(features[0][counter])
    # print (features)
    return features

def test_model (wav_file):
    # wav_file='k_close_2.wav'
    # wav_file='a_open_8.wav'

    wav_file='(4).wav'
    features=prepare_testing(wav_file)
    model= pickle.load(open('model_random2.pkl','rb'))
    model_output =model.predict(features)
    print (model_output[0])
    
    if   model_output[0]==0:
        result='Close the door'
    elif model_output[0]==1:
        result='Open the door'
    else :
        result='not a correct sentence'
  

        
    print('reeeeeeesult---------------------')
    print(result)
    return result

    
def predict_sound(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))
    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)

    features=[]

    features.append(np.concatenate((mfccs, chroma, mel, contrast, tonnetz),axis=0))
    open_model = pickle.load(open(".pkl",'rb'))
    result =open_model.predict(features)
    print(result)
    return result

def  visualize(file_name):

    fig,ax = plt.subplots(figsize=(6,6))
    ax=sns.set_style(style='darkgrid')
    sr, y = wavfile.read(file_name)
    # select left channel only
    y = y[:,0]
    # trim the first 125 seconds
    first = y[:int(sr*15)]

    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(first, Fs=sr)
    # plt.show()
    canvas=FigureCanvas(fig)
    img=io.BytesIO()
    fig.savefig(img, format='png')

    img.seek(0)
    # Embed the result in the html output.
    data = base64.b64encode(img.getbuffer()).decode("ascii")
    image_file_name='static/assets/images/result'+str(variables.counter)+'.jpg'
    plt.savefig(image_file_name)
    return f"<img src='data:image/png;base64,{data}'/>"
    # return img



@app.route("/", methods=["GET", "POST"])
def index():

        speech =''
        speaker =''
        file_name=''
        img=''

        y=[]
        sr=[]


        if request.method == "POST":
            variables.counter+=1
            # Sampling frequency
            frequency = 44400
            # Recording duration in seconds
            duration = 1.5
            # to record audio from
            # sound-device into a Numpy
            recording = sd.rec(int(duration * frequency),samplerate = frequency, channels = 2)
            # Wait for the audio to complete
            sd.wait()
            # using wavio to save the recording in .wav format
            # This will convert the NumPy array to an audio
            # file with the given sampling frequency
            file_name='result'+str(variables.counter)+'.wav'
            wv.write(file_name, recording, frequency, sampwidth=2)
            speech=test_model(file_name)
            speaker=predict_sound(file_name)
            # y, sr = librosa.load(file_name)
            img= visualize(file_name)
            img='static/assets/images/result'+str(variables.counter)+'.jpg'


        # return send_file(speech=speech,speaker=speaker,file_name=file_name,y=y,sr=sr)
        return render_template('index.html', speech=speech,speaker=speaker,file_name=file_name,y=y,sr=sr,img=img)



if __name__ == "__main__":
    app.run(debug=True, threaded=True)
