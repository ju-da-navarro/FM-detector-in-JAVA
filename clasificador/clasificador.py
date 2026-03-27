import numpy as np
from algoritmo.espectro import determinar

def clasificar_audio(audio, espectro_fm, espectro_wn):
    espectro_audio = determinar(audio, "espec")

    diff_fm = np.mean(np.abs(espectro_audio - espectro_fm))
    diff_wn = np.mean(np.abs(espectro_audio - espectro_wn))

    return "FM" if diff_fm < diff_wn else "Ruido Blanco"