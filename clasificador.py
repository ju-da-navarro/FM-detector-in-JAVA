import numpy as np
from espectro import espectro_promedio

def clasificar_audio(audio, espectro_fm, espectro_wn):
    espectro_audio = espectro_promedio([audio], "temp.npy")

    diff_fm = np.mean(np.abs(espectro_audio - espectro_fm))
    diff_wn = np.mean(np.abs(espectro_audio - espectro_wn))

    if diff_fm < diff_wn:
        return "FM"
    else:
        return "Ruido Blanco"