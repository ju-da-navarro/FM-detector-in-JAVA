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

    #---------- En caso de tener una lista (Generalmente es cuando se crean los patrones de referencia) -----------------
    if isinstance(audio_data, list):
        resultados = []

        for audio in audio_data:
            resultado = calc(audio, type)

            if type == "acov": #Tratamiento para autocovarianza
                if resultado[0] != 0:
                    resultado /= resultado[0]
                
                resultado = resultado[:2000]
             
            else: #Tratamiento para densidad espectral
                resultado = np.abs(resultado)
                if np.max(resultado) != 0: resultado /= np.max(resultado) #Normalización
            
            resultados.append(resultado)
        
        magnitud = np.mean(resultados, axis=0) #Promedio

        if file:
            np.save(file, magnitud)

        return magnitud
    
    #---------------- Tratamiento general para el audio del microfono -------------------
    
    res_indiv = calc(audio_data, type)

    if type == "acov": #Tratamiento para autocovarianza
        if res_indiv[0] != 0:
            res_indiv /= res_indiv[0]
        
        res_indiv = res_indiv[:2000]
    
    else: #Tratamiento para densidad espectral
        res_indiv = np.abs(res_indiv)

    return (res_indiv / np.max(res_indiv)) if np.max(res_indiv) != 0 else "Error matemático, división por 0" #Retornar con normalización
        
