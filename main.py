import os
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
from espectro import espectro_promedio
from clasificador import clasificar_audio

## UTILS

def cargar_audios(carpeta):
    audios = []

    for archivo in os.listdir(carpeta):
        if archivo.endswith(".wav"):
            ruta = os.path.join(carpeta, archivo)

            sr, audio = wavfile.read(ruta)
            audio = audio.astype(float)

            audios.append(audio)

    return audios


## GRABACION DE AUDIO SENCILLA

def grabar_audio(duracion=3, samplerate=44100):
    print("Grabando...")
    audio = sd.rec(int(duracion * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print("Grabación terminada")

    return audio.flatten()


## LOGICA DEL MAIN

if __name__ == "__main__":

    print("1. Entrenar modelo")
    print("2. Analizar audio en tiempo real")

    opcion = input("Seleccione opción: ")

    if opcion == "1":
        audios_fm = cargar_audios("data/FM")
        audios_wn = cargar_audios("data/WN")

        espectro_promedio(audios_fm, "espectro_FM.npy")
        espectro_promedio(audios_wn, "espectro_WN.npy")

        print("Modelo entrenado ✅")

    elif opcion == "2":

        # cargar espectros guardados
        espectro_fm = np.load("espectro_FM.npy")
        espectro_wn = np.load("espectro_WN.npy")

        audio = grabar_audio()

        resultado = clasificar_audio(audio, espectro_fm, espectro_wn)

        print("Resultado:", resultado)

    else:
        print("Opción inválida")