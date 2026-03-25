import numpy as np
from statsmodels.tsa.stattools import acf

LONGITUD = 44100*2 # 2 segundo

def recortar(audio):
    if len(audio) >= LONGITUD:
        return audio[:LONGITUD]
    else:
        return np.pad(audio, (0, LONGITUD - len(audio)))

#Función que calcula la diferencia con la media, la autocovarianza y la fft, retorna la fft
def calcularFFT(audio):

    #Asegurar audio mono para refinación de calculos
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    #Recortar el audio
    audio = recortar(audio)

    #Calcular la diferencia con la media
    x = audio - np.mean(audio)

    #Calcular autocovarianza
    autocov = acf(x, nlags=LONGITUD-1, fft=True)

    #Retornar la fft de la autocovarianza previamente calculada
    return np.fft.rfft(autocov)

#Función que procesa en caso de que se reciba una lista para entrenamiento o un único audio para clasificación en tiempo real
def determinar_espectro(audio_data, tipo = ""):

    #En caso de que procesemos una lista de audios (Generalmente es para entrenar)
    if isinstance(audio_data, list):
        espectros = []

        for audio in audio_data:
            espectros.append(np.abs(calcularFFT(audio)))

        #Calcular promedio
        espectro = np.mean(espectros, axis = 0)

        #Normalización
        espectro /= np.max(espectro)

        #Almacenar
        if tipo:
           np.save(tipo, espectro)
    
        return espectro
    
    #En caso de que se ingrese un valor único (Generalmente es para clasificar)
    espectro = np.abs(calcularFFT(audio_data))

    if tipo:
        np.save(tipo, espectro)

    return (espectro / np.max(espectro)) if (np.max(espectro) != 0) else espectro