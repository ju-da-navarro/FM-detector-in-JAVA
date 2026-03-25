import tkinter as tk
import sounddevice as sd
from clasificador import clasificar_audio
import numpy as np


## GRABACION DE AUDIO SENCILLA
def grabar_audio(duracion=2, samplerate=44100):
    print(f"Grabando...{duracion}s")
    audio = sd.rec(int(duracion * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print("Grabación terminada")

    return audio.flatten()


def inicioInterfaz():
    root = tk.Tk()
    root.title("Analizador de FM-WN")
    root.geometry("800x400")

    label = tk.Label(root, text="Interfaz Grafica simple")
    label.pack()

    label_resultado = tk.Label(root, text="Resultado aquí...")
    label_resultado.pack()

    # cargar espectros guardados
    espectro_fm = np.load("espectro_FM.npy")
    espectro_wn = np.load("espectro_WN.npy")
    
    def manejar_grabacion():
        audio = grabar_audio()  
        resultado = clasificar_audio(audio, espectro_fm, espectro_wn)
        label_resultado.config(text=f"El audio analizado es: {resultado}")

    boton = tk.Button(root, text="Grabar Audio", command=manejar_grabacion)
    boton.pack()

    root.mainloop()
