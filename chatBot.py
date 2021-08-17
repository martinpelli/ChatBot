# Info
# Al ejecutar muestra varias warnings, no es necesario prestarle atencion por el momento.


# ------------------------------------------------------------------------#
# >>>>>>>>>>>>>>>>>>>>> Subprograma mainBot <<<<<<<<<<<<<<<<<<<<<<<#

# Importa libreria que transforma palabras
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy
import tflearn
from tensorflow.python.framework import ops
import json
import random
from docplex.cp.model import CpoModel
import pickle



with open("contenido.json") as archivo:
    datos = json.load(archivo)

# print(datos) # Imprime los datos del archivo contenido.json

# Arrays donde se va a guardar la informacion
palabras = []
tags = []
auxX = []
auxY = []

# Itera sobre los datos de contenido.json
for contenido in datos["contenido"]:
    for patrones in contenido["patrones"]:
        auxPalabra = nltk.word_tokenize(patrones)  # Toma una frase y las separa en palabras (Reconoce . ? ! , )
        palabras.extend(auxPalabra)  # Insertamos las palabras separadas en el Array palabras
        auxX.append(auxPalabra)
        auxY.append(contenido["tag"])

        # En el Array tags no queremos que este el contenido repetido,
        # por lo que llevamos a cabo un if para verificar.
        if contenido["tag"] not in tags:  # Si este tag no esta en la lista vacia, se agrega
            tags.append(contenido["tag"])

# print(palabras) # Todas las palabras separadas
# print(auxX) # Lista de listas con distintas frases (con palabras separadas)
# print(auxY) # Son los tags repetidos
# print(tags) # Array de tags sin repetir


# Tranforma todas las palabras a minuscula para que sea mas entendible para el
# bot, ademas no toma en cuenta el signo de preguntas
palabras = [stemmer.stem(w.lower()) for w in palabras if w != '?']
# Ordenamos las palabras
palabras = sorted(list(set(palabras)))
tags = sorted(tags)

entrenamiento = []
salida = []

# Salida rellena de puros 0's con un rango de la cantidad de palabras de tags
salidaVacia = [0 for _ in range(len(tags))]

# Regresa la lista de palabras con un indice en cada una de ellas
# La palabra se guarda en documento y el indice en x

# Recorremos el documento que tiene insertado las palabras de auxX
# para rellenar la cubeta segun lo siguiente
for x, documento in enumerate(auxX):
    # La cubeta sirve para guardar 1's y 0's cuando la palabra esta
    # en auxPalabra y cuando no
    cubeta = []
    auxPalabra = [stemmer.stem(w.lower()) for w in documento]
    for w in palabras:
        if w in auxPalabra:
            cubeta.append(1)
        else:
            cubeta.append(0)

    filaSalida = salidaVacia[:]

    # Para recordar
    # 1. en el array tags estan los tags sin repetir
    # 2. en el array auxY estan los tags repetidos
    # 3. la posicion x es el indice que obtenemos en el for mediante el enumerate
    # Obtenemos el contenido que hay en auxY por cada uno de los indices de x
    # Por cada contenido, lo buscamos en la lista de tags y le asignamos un 1.
    filaSalida[tags.index(auxY[x])] = 1

    entrenamiento.append(cubeta)
    salida.append(filaSalida)

# print(entrenamiento) # Lista de listas con 1's y 0's, cada 1 representa que hay una palabra encontrada
# print(salida) # Lista de listas con 2 numeros, el primero es saludo y el segundo es despedida para saber a que tag nos estamos refiriendo

# Lo pasamos a arreglo de numpy
entrenamiento = numpy.array(entrenamiento)
salida = numpy.array(salida)

# Reinicia la red neuronal
ops.reset_default_graph()

red = tflearn.input_data(shape=[None, len(entrenamiento[0])])

# Fully_connected hace referencia a que cada entrada va a estar conectada
# con todas las neuronas de la siguiente columna

# Dos columnas de neuronas que representan las entradas
# Una columna de neuronas que representan las salidas
red = tflearn.fully_connected(red, 10)  # 10 Representa la cantidad de neuronas
red = tflearn.fully_connected(red, 10)

# Tiene la longitud de la cantidad de tags de nuestro arreglo de salidas
# El activation softmax nos permite hacer predicciones para catalogar las salidas
red = tflearn.fully_connected(red, len(salida[0]), activation="softmax")

# Nos permite obtener probabilidades respecto a que tan eficaz es nuestra
# clasificacion para saber a que tag nos estamos refiriendo
red = tflearn.regression(red)

# Tenemos que moldear nuestro modelo
modelo = tflearn.DNN(red)
# n_epoch hace referencia a cantidad de veces que va a estar viendo nuestro
# modelo los datos para entrenar al algoritmo sobre a que tag nos estamos
# refiriendo.
# batch_size esta relacionado con las entradas que vamos a ocupar de las
# neuronas.
modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=10, show_metric=True)
modelo.save("modelo.tflearn")


# Funcion encargada de responder en base a la entrada
def mainBot():
    while True:
        entrada = input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada)  # Separar las palabras entradas
        # Con esto hacemos que las palabras sean entendibles para el chatbot
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        for palabraIndividual in entradaProcesada:
            # En i guarda el indice del elemento y en palabra el elemento en si.
            for i, palabra in enumerate(palabras):
                # Verificamos si se han usado nuestras palabras en patrones
                if palabra == palabraIndividual:
                    # En esa posicion de la cubeta (indice) se esta usando esa palabra
                    cubeta[i] = 1

        # Ahora sacamos la probabilidad de a que tag nos estamos refiriendo
        # Si es a una despedida o a un saludo
        resultados = modelo.predict([numpy.array(cubeta)])

        # Imprime 2 numeros, el primero hace referencia al tag de despedida
        # El segundo hace referencia al tag de saludo
        # print(resultados)

        # Obtenemos el indice del tag el cual tiene mas probabilidad de ser
        # al que nos estamos refiriendo
        resultadosIndices = numpy.argmax(resultados)
        tag = tags[resultadosIndices]
        if tag == "opl":
            print("BOT: ")
            solveOPLModel()
        else:
            # Cargamos en respuestas las posibles respuestas que tenemos para
            # la entrada
            for tagAux in datos["contenido"]:
                if tagAux["tag"] == tag:
                    respuesta = tagAux["respuestas"]
            print("BOT: ", str(random.choice(respuesta)))

def solveOPLModel():
    mdl = CpoModel()
    masonry = mdl.interval_var(name='masonry', size=35)
    carpentry = mdl.interval_var(name='carpentry', size=15)
    plumbing = mdl.interval_var(name='plumbing', size=40)
    ceiling = mdl.interval_var(name='ceiling', size=15)
    roofing = mdl.interval_var(name='roofing', size=5)
    painting = mdl.interval_var(name='painting', size=10)
    windows = mdl.interval_var(name='windows', size=5)
    facade = mdl.interval_var(name='facade', size=10)
    garden = mdl.interval_var(name='garden', size=5)
    moving = mdl.interval_var(name='moving', size=5)

    # Add precedence constraints
    mdl.add(mdl.end_before_start(masonry, carpentry))
    mdl.add(mdl.end_before_start(masonry, plumbing))
    mdl.add(mdl.end_before_start(masonry, ceiling))
    mdl.add(mdl.end_before_start(carpentry, roofing))
    mdl.add(mdl.end_before_start(ceiling, painting))
    mdl.add(mdl.end_before_start(roofing, windows))
    mdl.add(mdl.end_before_start(roofing, facade))
    mdl.add(mdl.end_before_start(plumbing, facade))
    mdl.add(mdl.end_before_start(roofing, garden))
    mdl.add(mdl.end_before_start(plumbing, garden))
    mdl.add(mdl.end_before_start(windows, moving))
    mdl.add(mdl.end_before_start(facade, moving))
    mdl.add(mdl.end_before_start(garden, moving))
    mdl.add(mdl.end_before_start(painting, moving))

    # -----------------------------------------------------------------------------
    # Solve the model and display the result
    # -----------------------------------------------------------------------------

    # Solve model
    msol = mdl.solve(TimeLimit=10)
    return msol.print_solution()



mainBot()
