import speech_recognition as sr
import google_speech 
import os
import json
GOOGLE_APPLICATION_CREDENTIALS = os.path.join(os.getcwd(), 'Multimodal Interaction-d742de3e5198.json')

def speec_to_text():    
    """

    This function does not take any input and does speech recognition itself.
    sr.Recognizer() is the function of speech_recogniton that does the voice to text convertion
    sr.Microphone() is the function that uses the device microphone
    r.adjust_for_ambient_noise() adjustes the lower threshold of noise based on the environment noise
    r.listen() is the function that records a phrase and can wait up to 10 seconds if no phrase is said
    recognize_google() is the function that decides which recogniser we going to use and it works online


    """

    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("enters recording")
        r.adjust_for_ambient_noise(source)
        print("waiting for input")
        audio = r.listen(source, timeout=5)
      
        print("recognizing the input")
        
        # jsonCred = json.loads(GOOGLE_APPLICATION_CREDENTIALS)
    try:
        text = r.recognize_google(audio)
        print("You said: " + text)    # recognize speech using Google Speech Recognition
    except LookupError:                            # speech is unintelligible
        
        print("Could not understand audio")

    return text


def text_to_spech(text="Hello world"):
    """
    Text-To-Speech from its name takes input a string or a text and transforms it into audio
    .Speech functio is google function that takes as input the text and reads it, can work on and ofline
    """
    sox_effect1 = ("delay", "0.5")
    sox_effect2 = ()
    text_to_spech = google_speech.Speech(text, 'en')
    text_to_spech.play(sox_effect1)

    return


# text = speec_to_text()
# text_to_spech(text)




