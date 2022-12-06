from flask import Flask, render_template, request, redirect,jsonify
import os


app = Flask(__name__)


@app.route("/receive", methods=['post'])
def form():
    files = request.files
    file = files.get('file')
    print(file)

    with open(os.path.abspath(f'backend/audios/{file}'), 'wb') as f:
        f.write(file.content)

    response = jsonify("File received and saved!")
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

# import speech_recognition as sr



# @app.route("/", methods=["GET", "POST"])
# def index():


#         speech=''
#         state=''
#         source=''

#         if request.method == "POST":
#             r = sr.Recognizer()

#             with sr.Microphone() as source:
#                 print("Password")

#                 r.adjust_for_ambient_noise(source)
#                 audio = r.listen(source)

#                 print("Recognizing Now .... ")


#                 # recognize speech using google
#                 try:
#                     speech=r.recognize_google(audio)
#                     print("You have said \n" + speech)
#                 except Exception as e:
#                     print("Error :  " + str(e))
                    
#                 # Saving audio
#                 with open("recorded.wav", "wb") as f:
#                     f.write(audio.get_wav_data())

#                 if (speech=='open the door'):
#                     state='successfully opened'
#                     print ('successfully opened')
                
#                 elif (speech=='close the door'):
#                     state='successfully closed'
#                     print ('successfully closed')
#                 else :
#                     state='not a correct pass'
#                     print('not a correct pass')
#         return render_template('index.html', speech=speech,state=state )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
