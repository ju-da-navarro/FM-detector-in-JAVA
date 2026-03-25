import os
import numpy as np
from scipy.io import wavfile
from espectro import determinar_espectro
from interfaz import inicioInterfaz

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

## LOGICA DEL MAIN

if __name__ == "__main__":

    print("1. Entrenar modelo")
    print("2. Analizar audio en tiempo real")

    opcion = input("Seleccione opción: ")

    if opcion == "1":
        audios_fm = cargar_audios("data/FM")
        audios_wn = cargar_audios("data/WN")

        determinar_espectro(audios_fm, "espectro_FM.npy")
        determinar_espectro(audios_wn, "espectro_WN.npy")

        print("Modelo entrenado Correctamente")

    elif opcion == "2":
        inicioInterfaz()
        print("Programa finalizado exitosamente")

    else:
        print("Opción inválida")