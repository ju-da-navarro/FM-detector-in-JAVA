import numpy as np
from statsmodels.tsa.stattools import acovf

LONGITUD = 44100*2
    
def calc(audio, type):
    if (type != "acov" and type != "espec"): return "Tipo incorrecto"

    #Asegurar mono
    if len(audio.shape) > 1: audio = np.mean(audio, axis = 1)
    x = audio - np.mean(audio)
    autocov = acovf(x, fft=True, demean=True).astype(np.float64)

    return autocov if type == "acov" else np.fft.rfft(autocov)


def determinar(audio_data, type, file=""):
    if (type != "acov" and type != "espec"): return "Tipo incorrecto"

    if isinstance(audio_data, list):
        espectros = []

        for audio in audio_data:
            espectros.append(np.abs(calc(audio, type)))
        
        espectro = np.mean(espectros, axis = 0)
        espectro /= np.max(espectro)

        if file:
            np.save(file, espectro)
        
        return espectro
    
    espectro = np.abs(calc(audio_data, type))
    return (espectro / np.max(espectro)) if np.max(espectro) != 0 else espectro
        
