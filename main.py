from flask import Flask, render_template, request, redirect
import speech_recognition as sr
class variables:
    button_pressed_num=0

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

        speech=''
        state=''
        source=''

        if request.method == "POST":
            variables.button_pressed_num+=1
            print( variables.button_pressed_num)

            r = sr.Recognizer()

            with sr.Microphone() as source:
                print("Password")
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                
                print("Recognizing Now .... ")


                # recognize speech using google
                try:
                    speech=r.recognize_google(audio)
                    print("You have said \n" + speech)
                except Exception as e:
                    print("Error :  " + str(e))
                    
                # Saving audio
                with open("recorded.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                if (speech=='open the door'):
                    state='successfully opened'
                    print ('successfully opened')
                
                elif (speech=='close the door'):
                    state='successfully closed'
                    print ('successfully closed')
                else :
                    state='not a correct pass'
                    print('not a correct pass')
                
        return render_template('index.html', speech=speech,state=state )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)











