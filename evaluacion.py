import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score


##############################
# CODIGO PARA EVALUAR MODELOS
#################

def matriz_confusion(y_val, y_pred,etiquetas_clases=None,dosclases = True):

    if etiquetas_clases is None:
        etiquetas_clases = {0: "HU", 1: "IA"}

    cm = confusion_matrix(y_val, y_pred)

    clases = np.unique(y_val)
    labels = [etiquetas_clases[c] for c in clases]

    if dosclases: # igual que en texto -> azul
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")

    else: # multiclase 
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)

    # plt.xlabel("Predicho")
    # plt.ylabel("Real")
    # plt.title("Matriz de Confusión")
    # plt.show()
    tn, fp, fn, tp = cm.ravel()

    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

#######################################################################

def busqueda_mejor_umbral(modelo,X_val,y_val,umbrales=[0.5, 0.6, 0.7, 0.8, 0.9]):
    '''
    Busca el mejor umbral sobre el conjunto de Validación para un modelo dentro del rango 0-1 buscando maximizar el valor de F1
    '''
    y_proba = modelo.predict_proba(X_val)[:, 1]  # probabilidad de ser IA (clase positiva)

    mejor_umbral = None
    mejor_f1 = -1

    for umbral in umbrales:
        y_pred = (y_proba >= umbral).astype(int)
        f1 = f1_score(y_val, y_pred)

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral

    return mejor_umbral, mejor_f1

#######################################################################

def visualizar_confianza_vs_acierto(modelo, X_val, y_val,th=0.5):

    proba = modelo.predict_proba(X_val)
    y_proba = proba[:, 1]
    y_pred = (y_proba >= th).astype(int)
    max_proba = np.where(y_pred == 1, y_proba, 1 - y_proba)
    

    plt.hist(max_proba, bins=20)
    plt.xlabel("Probabilidad máxima")
    plt.ylabel("Muestras")
    plt.title("Confianza del clasificador")
    plt.show()

    correctos = max_proba[y_pred == y_val]
    incorrectos = max_proba[y_pred != y_val]

    plt.hist(correctos, bins=20, alpha=0.6, label='Correctos')
    plt.hist(incorrectos, bins=20, alpha=0.6, label='Incorrectos')
    plt.xlabel("Confianza")
    plt.ylabel("Muestras")
    plt.legend()
    plt.title("Confianza vs acierto")
    plt.show()

#######################################################################

def visualizar_confianza_vs_acierto_por_clase(modelo, X_val, y_val,th=0.5):

    proba = modelo.predict_proba(X_val)
    y_proba = proba[:, 1]
    y_pred = (y_proba >= th).astype(int)

    # IA = 1
    mask_correctos_ia = (y_val == 1) & (y_pred == 1)
    mask_incorrectos_ia = (y_val == 1) & (y_pred == 0)

    # HU = 0
    mask_correctos_hu = (y_val == 0) & (y_pred == 0)
    mask_incorrectos_hu = (y_val == 0) & (y_pred == 1)

    correctos_ia = y_proba[mask_correctos_ia]
    incorrectos_ia = y_proba[mask_incorrectos_ia]

    correctos_hu = 1 - y_proba[mask_correctos_hu]
    incorrectos_hu = 1 - y_proba[mask_incorrectos_hu]

    plt.hist(correctos_ia, bins=20, alpha=0.6, label='Correctos')
    plt.hist(incorrectos_ia, bins=20, alpha=0.6, label='Incorrectos')

    plt.xlabel("Confianza")
    plt.ylabel("Muestras")
    plt.legend()
    plt.title("Confianza vs acierto Muestras IA")
    plt.show()

    plt.hist(correctos_hu, bins=20, alpha=0.6, label='Correctos')
    plt.hist(incorrectos_hu, bins=20, alpha=0.6, label='Incorrectos')

    plt.xlabel("Confianza")
    plt.ylabel("Muestras")
    plt.legend()

    plt.title("Confianza vs acierto Muestras HU")

    plt.show()