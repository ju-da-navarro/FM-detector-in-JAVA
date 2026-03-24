import numpy as np
from statsmodels.tsa.stattools import acf

LONGITUD = 44100*3 # 1 segundo


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
        fft = np.fft.fft(autocov)
        espectros.append(np.abs(fft))

    espectro = np.mean(espectros, axis=0)

    # persistencia
    np.save(tipo, espectro)

    return espectro