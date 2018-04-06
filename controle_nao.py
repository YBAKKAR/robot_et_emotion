import naoqi as nao
import sys

module ="ALTextToSpeech"
ip_adress = "192.168.1.4"
port = 9559
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def saySomething(emotion):
    tts = nao.ALProxy(module,ip_adress,port)
    if emotion == emotions[0]:
        words = "Mais pourquoi tu es en colaire ?"
    elif emotion == emotions[1]:
        words = "Ughhh"
    elif emotion == emotions[2]:
        words = "Tu es un centralien tu doit rien craindre !"
    elif emotion == emotions[3]:
        words = "Oh, tu as l'air d'etre heureux aujourd'hui"
    elif emotion == emotions[4]:
        words = "Oups, pourquoi tu es triste?"
    elif emotion == emotions[5]:
        words = "Est-ce que je t'ai surpris?"
    elif emotion == emotions[6]:
        words = "bof, tu as un visage froid"

    tts.say(words)

saySomething(sys.argv[1])