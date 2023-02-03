#from speech_synthesis import synthesis_speech
import openai 
import speech_recognition as sr
from dotenv import load_dotenv
import os
import azure.cognitiveservices.speech as speechsdk


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
speech_key, speech_region = os.getenv("SPEECH_KEY"), os.getenv("SPEECH_REGION")

def synthesis_speech(text):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    speech_config.speech_synthesis_voice_name="en-US-JennyNeural"

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config = speech_config, audio_config = audio_config)

    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()



if __name__ == "__main__":
    
    # Prep microphone
    r = sr.Recognizer()
    r.energy_threshold = 200
    r.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        print("How would you like AI to help you? Speak, I am listening...")
        audio = r.listen(source, timeout = 5)

    
    user_question = r.recognize_google(audio, show_all=False, with_confidence=False)
    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=user_question,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    response_text = response.choices[0].text

    print(response_text)
    synthesis_speech(response_text)
