from flask import Flask, render_template, request, redirect,jsonify
import os
import speech_recognition as sr
import pickle
import librosa
import numpy as np
import sounddevice as sd
import wavio as wv



app = Flask(__name__)


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
    y, sr = librosa.load(to_test)
    # remove leading and trailing silence
    y, index    = librosa.effects.trim(y)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse        = librosa.feature.rms(y=y)
    spec_cent   = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw     = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(y)
    mfcc        = librosa.feature.mfcc(y=y, sr=sr)
    to_append   = f' {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    
    features.append(to_append.split())

    
    for index in range(0,len(features[0])):
        features[0][index]=float(features[0][index])

    return features

def test_model (wav_file):
    # wav_file='k_close_2.wav'
    
    features=prepare_testing(wav_file)
    model= pickle.load(open('model.pkl','rb'))
    model_output =model.predict(features)

    if model_output==0:
        result='Close the door'
    elif model_output==1:
        result='Open the door'
    else:
        result=''
        
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
    open_model = pickle.load(open("audio_identefire.pkl",'rb'))
    result =open_model.predict(features)
    print(result)
    return result



@app.route("/", methods=["GET", "POST"])
def index():


        speech =''
        speaker =''
        file_name=''
        y=[]
        sr=[]


        if request.method == "POST":

            # Sampling frequency
            frequency = 44400
            # Recording duration in seconds
            duration = 1
            # to record audio from
            # sound-device into a Numpy
            recording = sd.rec(int(duration * frequency),samplerate = frequency, channels = 2)
            # Wait for the audio to complete
            sd.wait()
            # using wavio to save the recording in .wav format
            # This will convert the NumPy array to an audio
            # file with the given sampling frequency
            wv.write("result.wav", recording, frequency, sampwidth=2)
            speech=test_model("result.wav")
            speaker=predict_sound("result.wav")
            file_name="result.wav"
            y, sr = librosa.load(file_name)




        return render_template('index.html', speech=speech,speaker=speaker,file_name=file_name,y=y,sr=sr)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
