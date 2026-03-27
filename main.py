import os
import numpy as np
import librosa
from algoritmo.espectro import determinar
from gui.interfaz import inicioInterfaz

## UTILS

def obtener_archivos(directorio):
    return [os.path.join(directorio, file) for file in os.listdir(directorio)]

def cargar_audio(archivo, freq = 44100):
    audio, _ = librosa.load(archivo, sr=freq, dtype=np.float64) 
    audio = librosa.util.fix_length(audio, size=freq*2)
    return audio

def cargarListaAudios(directorio, freq = 44100):
    archivos = obtener_archivos(directorio)
    audios = []

    for archivo in archivos:
        audio = cargar_audio(archivo, freq=freq)
        audios.append(audio)

    if not audios:
        print("Error, no se cargaron audios")
    else:
        print("Audios cargados correctamente")

    return audios

## LOGICA DEL MAIN

if __name__ == "__main__":

    print("1. Entrenar modelo")
    print("2. Analizar audio en tiempo real")

    opcion = input("Seleccione opción: ")

    if opcion == "1":
        audios_fm = cargarListaAudios("data/FM")
        audios_wn = cargarListaAudios("data/WN")

        determinar(audios_fm, "espec", "patrones_referencia/espectro_FM.npy")
        determinar(audios_wn, "espec", "patrones_referencia/espectro_WN.npy")
        determinar(audios_fm, "acov", "patrones_referencia/autocov_fm.npy")
        determinar(audios_wn, "acov", "patrones_referencia/autocov_wn.npy")

        print("Modelo entrenado Correctamente")

    elif opcion == "2":
        inicioInterfaz()
        print("Programa finalizado exitosamente")

    else:
        print("Opción inválida")