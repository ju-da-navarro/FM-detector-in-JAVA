import tkinter as tk
import sounddevice as sd
from clasificador.clasificador import clasificar_audio
from algoritmo.espectro import determinar, LONGITUD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

## GRABACION DE AUDIO SENCILLA
def grabar_audio(duracion=2, samplerate=44100):
    print(f"Grabando...{duracion}s")
    audio = sd.rec(int(duracion * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print("Grabación terminada")    

    return audio.flatten()

def inicioInterfaz():
    root = tk.Tk()
    root.title("Reconocedor de audio")
    titulo = tk.Label(root, text="Analizador de FM-WN", font=("Arial", 20, "bold"))
    titulo.pack()
    root.geometry("800x400")

    boton = tk.Button(root, text="Grabar Audio")
    boton.pack()

    label_resultado = tk.Label(
        root,
        text="Resultado aquí...",
        font=("Arial", 16, "bold"),
        fg="black"
    )
    label_resultado.pack()

    # cargar espectros guardados
    espectro_fm = np.load("patrones_referencia/espectro_FM.npy")
    espectro_wn = np.load("patrones_referencia/espectro_WN.npy")

    autocov_fm = np.load("patrones_referencia/autocov_fm.npy")
    autocov_wn = np.load("patrones_referencia/autocov_wn.npy")

    frecuencias = np.fft.rfftfreq(LONGITUD, d=1/44100)
    lags = np.arange(len(autocov_wn)) #Por esto fallaba, para graficar x se necesitaban era lags (las muestras)

    ##LOGICA DE ESPECTRO

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(frecuencias, espectro_wn, label="WN", color="blue")
    ax1.set_title("Espectro WN")
    ax1.set_xlabel("Frecuencia (Hz)")
    ax1.set_ylabel("Magnitud espectro")
    ax1.legend()

    ax2.plot(frecuencias, espectro_fm, label="FM", color="green")
    ax2.set_title("Espectro FM")
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("Magnitud espectro")
    ax2.legend()

    ax1.set_xlim(0, 500)
    ax2.set_xlim(0, 500)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    ##LOGICA DE AUTOCOV

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))

    ax3.plot(lags, autocov_wn, label="autocov WN", color="blue")
    ax3.set_title("Autocovarianza WN")
    ax3.set_xlabel("Lags (Muestras)")
    ax3.set_ylabel("Autocovarianza")
    ax3.legend()

    ax4.plot(lags, autocov_fm, label="autocov FM", color="green")
    ax4.set_title("Autocovarianza FM")
    ax4.set_xlabel("Lags (Muestras)")
    ax4.set_ylabel("Autocovarianza")
    ax4.legend()

    ax3.set_xlim(0, 500)
    ax4.set_xlim(0, 500)

    canvas2 = FigureCanvasTkAgg(fig2, master=root)
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    historial_espectro = []
    
    def manejar_grabacion():
        audio = grabar_audio()  
        resultado = clasificar_audio(audio, espectro_fm, espectro_wn)
        espectro_au = determinar(audio, "espec")
        autocov_au = determinar(audio, "acov")

        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        
        color = "red"

        ax1.plot(frecuencias, espectro_wn, label="WN", color="blue")
        ax1.set_title("Espectro WN")
        ax1.set_title("Espectro WN")
        ax1.set_xlabel("Frecuencia (Hz)")
        ax1.set_xlim(0, 500)

        ax2.plot(frecuencias, espectro_fm, label="FM", color="green")
        ax2.set_title("Espectro FM")
        ax2.set_title("Espectro WN")
        ax2.set_xlabel("Frecuencia (Hz)")
        ax2.set_xlim(0, 500)

        ax3.plot(lags, autocov_wn, label="autocov WN", color="blue")
        ax3.set_title("Autocovarianza WN")
        ax3.set_xlabel("Lags (Muestras)")
        ax3.set_ylabel("Autocovarianza")
        ax3.set_xlim(0, 500)

        ax4.plot(lags, autocov_fm, label="autocov FM", color="green")
        ax4.set_title("Autocovarianza FM")
        ax4.set_xlabel("Lags (Muestras)")
        ax4.set_ylabel("Autocovarianza")
        ax4.set_xlim(0, 500)

        # añadir a ambas gráficas
        ax1.plot(frecuencias, espectro_au, linestyle="--", alpha=0.4, color=color, label="Audio")
        ax2.plot(frecuencias, espectro_au, linestyle="--", alpha=0.4, color=color, label="Audio")

        ax3.plot(lags, autocov_au, linestyle="--", alpha=0.4, color=color, label="Audio")
        ax4.plot(lags, autocov_au, linestyle="--", alpha=0.4, color=color, label="Audio")

        ax1.legend()
        ax2.legend()
        canvas.draw()  

        ax3.legend()
        ax4.legend()
        canvas2.draw()

        label_resultado.config(text=f"El audio analizado es: {resultado}", fg="blue" if resultado == "Ruido Blanco" else "green")

        historial_espectro.append(espectro_au)
    
    boton['command'] = manejar_grabacion

    def cerrar_app():
        print("Guardando datos...")
        sd.stop()
        np.save("patrones_referencia/micEspec.npy", historial_espectro)
        root.quit()

    root.protocol("WM_DELETE_WINDOW", cerrar_app)
    root.mainloop()
    root.destroy()
    
