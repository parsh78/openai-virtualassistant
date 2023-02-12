import openai 
import speech_recognition as sr
from dotenv import load_dotenv
import os
import azure.cognitiveservices.speech as speechsdk
import keyboard 

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
speech_key, speech_region = os.getenv("SPEECH_KEY"), os.getenv("SPEECH_REGION")

#Uses Azure Cognitive Services to convert text to natural sounding speech
def synthesize_speech(text):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name="en-GB-RyanNeural"
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config = speech_config, audio_config = audio_config)
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()


def prompt_to_listen(recognizer, prompt):
    print(prompt)

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        audio = recognizer.listen(source, timeout = 10)
    try:
        #user_question = recognizer.recognize_google(audio_data=audio, pfilter=0, show_all=False, with_confidence=False)
        user_question = recognizer.recognize_google(audio_data=audio)
        print(f"\n<< {user_question.capitalize()}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    if len(user_question) > 0:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=user_question,
            temperature=0,
            max_tokens=250, #You can go upto 4000, it impacts cost
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        response_text = response.choices[0].text
        print(f"{response_text}")
        print(f"\nTotal Tokens Consumed: {response.usage.total_tokens}")
        synthesize_speech(response_text)

        action_prompt = f"\nPress Spacebar to ask follow up question? Press Enter to end."
        print(action_prompt)

        while True:
            if keyboard.read_key() == "space":
                prompt = f"\nGo ahead, I am listening..."
                prompt_to_listen(recognizer, prompt)
                break
            if keyboard.read_key() == "enter":
                break


if __name__ == "__main__":
    # Prep microphone
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 200
    recognizer.dynamic_energy_threshold = True
    
    context_prompt = f"\nHow would you like AI to help you? Speak, I am listening..."
    
    # Recursive method to process user input with OpenAI API
    prompt_to_listen(recognizer, context_prompt)