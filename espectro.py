import numpy as np
from statsmodels.tsa.stattools import acf

LONGITUD = 44100*2 # 2 segundo


def recortar(audio):
    if len(audio) >= LONGITUD:
        return audio[:LONGITUD]
    else:
        return np.pad(audio, (0, LONGITUD - len(audio)))


def espectro_promedio(audios, tipo):
    espectros = []

    for audio in audios:

        # asegurar 1D (mono)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # recorte
        audio = recortar(audio)

        x = audio - np.mean(audio)

        # autocovarianza con acf (rápido con FFT)
        autocov = acf(x, nlags=LONGITUD-1, fft=True)

        # FFT + magnitud
        fft = np.fft.rfft(autocov)  
        espectros.append(np.abs(fft))

    espectro = np.mean(espectros, axis=0)

    #normalizar, opcional pero mejor
    espectro = espectro / np.max(espectro)
    
    # persistencia
    np.save(tipo, espectro)

    return espectro


def espectro_individual(audio):
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = recortar(audio)

    x = audio - np.mean(audio)

    autocov = acf(x, nlags=LONGITUD-1, fft=True)

    espectro = np.abs(np.fft.rfft(autocov))

    if np.max(espectro) != 0:
        espectro = espectro / np.max(espectro)

    return espectro