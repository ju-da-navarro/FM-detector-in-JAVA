import tkinter as tk
import sounddevice as sd
from clasificador import clasificar_audio
from espectro import LONGITUD, determinar_espectro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys



## GRABACION DE AUDIO SENCILLA
def grabar_audio(duracion=2, samplerate=44100):
    print(f"Grabando...{duracion}s")
    audio = sd.rec(int(duracion * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print("Grabación terminada")

    return audio.flatten()


def inicioInterfaz():
    root = tk.Tk()
    titulo = tk.Label(root, text="Analizador de FM-WN", font=("Arial", 20, "bold"))
    titulo.pack()
    root.geometry("800x400")

    label = tk.Label(root, text="Interfaz Grafica simple")
    label.pack()

    boton = tk.Button(root, text="Grabar Audio")
    boton.pack()

    label_resultado = tk.Label(
    root,
    text="Resultado aquí...",
    font=("Arial", 16, "bold"),
    fg="blue"
    )
    label_resultado.pack()

    # cargar espectros guardados
    espectro_fm = np.load("espectro_FM.npy")
    espectro_wn = np.load("espectro_WN.npy")

    frecuencias = np.fft.rfftfreq(LONGITUD, d=1/44100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # --- WN ---
    ax1.plot(frecuencias, espectro_wn, label="WN", color="blue")
    ax1.set_title("Espectro WN")
    ax1.legend()

    # --- FM ---
    ax2.plot(frecuencias, espectro_fm, label="FM", color="green")
    ax2.set_title("Espectro FM")
    ax2.legend()

    ax1.set_xlim(0, 4000)
    ax2.set_xlim(0, 4000)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    historial_espectro = []
    
    def manejar_grabacion():
        audio = grabar_audio()  
        resultado = clasificar_audio(audio, espectro_fm, espectro_wn)
        espectro_au = determinar_espectro(audio)

        ax1.cla()
        ax2.cla()
        
        color = "red"

        ax1.plot(frecuencias, espectro_wn, label="WN", color="blue")
        ax1.set_title("Espectro WN")
        ax1.set_xlim(0, 4000)

        ax2.plot(frecuencias, espectro_fm, label="FM", color="green")
        ax2.set_title("Espectro FM")
        ax2.set_xlim(0, 4000)

        # añadir a ambas gráficas
        ax1.plot(frecuencias, espectro_au, linestyle="--", alpha=0.4, color=color, label="Audio")
        ax2.plot(frecuencias, espectro_au, linestyle="--", alpha=0.4, color=color, label="Audio")

        ax1.legend()
        ax2.legend()

        canvas.draw()  
        label_resultado.config(text=f"El audio analizado es: {resultado}")

        historial_espectro.append(espectro_au)
    
    boton['command'] = manejar_grabacion

    def cerrar_app():
        print("Guardando datos...")
        sd.stop()
        np.save("temp.npy", historial_espectro)
        root.quit()

    root.protocol("WM_DELETE_WINDOW", cerrar_app)
    root.mainloop()
    root.destroy()
    
