import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 

warnings.filterwarnings("ignore")

def calculate_delta(array):

    rows,cols = array.shape
    print(rows)
    print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    	 
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)

    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def record_audio_test():

	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	CHUNK = 512
	RECORD_SECONDS = 2.5
	device_index = 2
	audio = pyaudio.PyAudio()
	index=1
	print()
	stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,input_device_index = index,
						frames_per_buffer=CHUNK)
	print ("recording started")
	Recordframes = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			Recordframes.append(data)
	print ("recording stopped")
	stream.stop_stream()
	stream.close()
	audio.terminate()

	OUTPUT_FILENAME="sample.wav"
	WAVE_OUTPUT_FILENAME=os.path.join("testing_set",OUTPUT_FILENAME)
	trainedfilelist = open("testing_set_addition.txt", 'a')
	trainedfilelist.write(OUTPUT_FILENAME+"\n")
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(Recordframes))
	waveFile.close()

def test_model():

	modelpath 	= "trained_models\\"


	gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

	#Load the Gaussian gender Models
	models     = [pickle.load(open(fname,'rb')) for fname in gmm_files]
	speakers   = [fname.split("\\")[-1].split(".gmm")[0]  for fname in gmm_files]
	
	# # Read the test directory and get the list of test audio files 
	# for path in file_paths:   
	# path = file_paths[0]
	# path = path.strip()   
	# print(path)

	# sr,audio = read(source + path)
	sr,audio = read('testing_set/sample.wav')
	vector   = extract_features(audio,sr)

	log_likelihood = np.zeros(len(models)) 

	for i in range(len(models)):
		gmm    = models[i]  #checking with each model one by one
		scores = np.array(gmm.score(vector))
		log_likelihood[i] = scores.sum()


	winner = np.argmax(log_likelihood)
	time.sleep(1.0) 
	score =max(abs(log_likelihood))
	return  speakers[winner] , log_likelihood
#choice=int(input("\n1.Record audio for training \n 2.Train Model \n 3.Record audio for testing \n 4.Test Model\n"))

def draw_bars(score):
    speaker_scores = [0, 0, 0, 0]
    speaker_scores[0] = abs((score[0]+score[1]+score[2]+score[3])/4)
    speaker_scores[1] = abs((score[4]+score[5]+score[6]+score[7])/4)
    speaker_scores[2] = abs((score[8]+score[9]+score[10]+score[11])/4)
    speaker_scores[3] = abs((score[12]+score[13]+score[14]+score[15])/4)
    sentence_scores = [0, 0, 0, 0]
    sentence_scores[0] = abs((score[0]+score[4]+score[8]+score[12])/4)
    sentence_scores[1] = abs((score[1]+score[5]+score[9]+score[13])/4)
    sentence_scores[2] = abs((score[2]+score[6]+score[10]+score[14])/4)
    sentence_scores[3] = abs((score[3]+score[7]+score[11]+score[15])/4)

    return speaker_scores, sentence_scores


def start_testing():
    result1 = ''
    result2 = ''
    record_audio_test()
    selected_model, score = test_model()
    y1, y2 = draw_bars(score)
    print('socre', score)
    print('max score', max(score))
    if (max(score) < -28):
        result1 = 'others'
    elif(selected_model == 'mariam' or selected_model == 'mariam2' or selected_model == 'mariam3' or selected_model == 'mariam4'):
        result1 = 'Mariam'
    elif(selected_model == 'abram' or selected_model == 'abram2' or selected_model == 'abram3' or selected_model == 'abram4'):
        result1 = 'Abram'
    elif(selected_model == 'naira' or selected_model == 'naira2' or selected_model == 'naira3' or selected_model == 'naira4'):
        result1 = 'Naira'
    elif(selected_model == 'hager' or selected_model == 'hager2' or selected_model == 'hager3' or selected_model == 'hager4'):
        result1 = 'Hager'

    if(selected_model == 'mariam' or selected_model == 'abram' or selected_model == 'naira' or selected_model == 'hager'):
        result2 = 'Open the door'
    else:
        result2 = 'Not a correct sentence      '

    return result1, result2, y1, y2 ,score



