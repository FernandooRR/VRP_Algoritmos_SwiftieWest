import sys
import random
import math
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox
import itertools
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk


def limpiarPantalla():
    for widget in app.grid_slaves():
        widget.grid_forget()

def menu():
    limpiarPantalla()
    Titulo.grid(row=0, column=1, padx=60, pady=40, sticky="nsew")
    fuerzaBrutaBoton.grid(row=1, column=1, pady=5)
    divideVencerasBoton.grid(row=2, column=1, pady=5)
    dinamicaBoton.grid(row=3, column=1, pady=5)
    greedyBoton.grid(row=4, column=1, pady=5)
    metaBoton.grid(row=5, column=1, pady=5)

def fuerzaBruta():
    limpiarPantalla()

    fuerzaTitulo = ctk.CTkLabel(app, text="PROBLEMA DEL VIAJANTE DE COMERCIO (TSP)", font=("Helvetica", 20))
    fuerzaTitulo.grid(row=0, column=0, columnspan=4, padx=20, pady=20, sticky="ew")
    explicacion_texto = "Genera todas las posibles permutaciones de las ubicaciones que deben\n"\
                        "visitarse (excluyendo la inicial), calcula la distancia total para cada\n" \
                        "ruta posible que comienza y termina en la ubicación inicial y finalmente,\n" \
                        "selecciona la ruta con la distancia mínima.\n" \
                        "Este enfoque asegura encontrar la solución óptima, pero no es eficiente \n" \
                        "para grandes cantidades de ubicaciones debido a la explosión\n" \
                        "combinatoria del número de permutaciones."
    explicacion = ctk.CTkLabel(app, text=explicacion_texto)
    explicacion.grid(row=1, column=0, columnspan=4, padx=10, pady=20, sticky="ew")

    # Función para calcular la distancia total de una ruta
    def calcularDistancia(ruta, matrizDistancia):
        distanciaTotal = 0
        for i in range(len(ruta) - 1):
            distanciaTotal += matrizDistancia[ruta[i]][ruta[i+1]]
        distanciaTotal += matrizDistancia[ruta[-1]][ruta[0]]
        return distanciaTotal

    # Función para resolver el problema del TSP mediante fuerza bruta
    def fuerzaBrutaVRP(matrizDistancia):
        numUbicaciones = len(matrizDistancia)
        ubicaciones = list(range(1, numUbicaciones))
        mejorRuta = None
        minimaDistancia = float('inf')

        distancias = []
        for rutas in itertools.permutations(ubicaciones):
            distancia = calcularDistancia([0] + list(rutas) + [0], matrizDistancia)
            distancias.append(distancia)
            if distancia < minimaDistancia:
                minimaDistancia = distancia
                mejorRuta = rutas

        return mejorRuta, minimaDistancia, distancias

    # Función para obtener la matriz de distancias ingresada por el usuario
    def obtenerMatrizDistancia():
        try:
            matriz = []
            for i in range(4):
                fila = []
                for j in range(4):
                    entry = float(entry_matrix[i][j].get())
                    fila.append(entry)
                matriz.append(fila)
            return matriz
        except ValueError:
            messagebox.showerror("Error", "Introduce números válidos, por favor.")
            return None

    # Función para resolver el problema y mostrar los resultados en la interfaz gráfica
    def solucionFuerzaBruta():
        matrizDistancia = obtenerMatrizDistancia()
        if matrizDistancia:
            mejorRuta, minimaDistancia, distancias = fuerzaBrutaVRP(matrizDistancia)

            mejorRutaText = "Mejor ruta: " + str(mejorRuta)
            mejorRutaTextLabel = ctk.CTkLabel(app, text=mejorRutaText)
            mejorRutaTextLabel.grid(row=6, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

            minimaDistanciaText = "Distancia mínima: " + str(minimaDistancia)
            minimaDistanciaTextLabel = ctk.CTkLabel(app, text=minimaDistanciaText)
            minimaDistanciaTextLabel.grid(row=7, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

            plt.figure(figsize=(10, 5))
            plt.plot(distancias, marker='o')
            plt.title('Distancias de rutas generadas por Fuerza Bruta')
            plt.xlabel('Iteración')
            plt.ylabel('Distancia')
            plt.grid(True)
            plt.show()

    # Creación de la interfaz gráfica para ingresar la matriz de distancias
    matrizDistanciaLabels = []
    entry_matrix = []

    for i in range(4):
        row_labels = []
        row_entries = []
        for j in range(4):
            entry = ctk.CTkEntry(app, width=50)
            entry.grid(row=2+i, column=j, padx=5, pady=5)
            row_entries.append(entry)
        matrizDistanciaLabels.append(row_labels)
        entry_matrix.append(row_entries)

    # Botón para resolver el problema y mostrar los resultados
    resolverBoton = ctk.CTkButton(app, text="Resolver", command=solucionFuerzaBruta)
    resolverBoton.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

    # Botón para regresar al menú principal
    regresarBoton = ctk.CTkButton(app, text="Regresar", command=menu)
    regresarBoton.grid(row=9, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

def divideVenceras():
    limpiarPantalla()

    divideTitulo = ctk.CTkLabel(app, text="PROBLEMA DEL VIAJANTE DE COMERCIO (TSP)", font=("Helvetica", 20))
    divideTitulo.grid(row=0, column=0, columnspan=4, padx=20, pady=20, sticky="ew")
    explicacion_texto = "Este enfoque divide recursivamente el conjunto de ubicaciones en dos\n"\
                        "subgrupos hasta que los subgrupos contienen dos o menos ubicaciones. \n" \
                        "Luego, calcula las rutas y distancias para estos subgrupos y las combina\n" \
                        "para obtener una ruta y distancia total. Proporciona una aproximación\n" \
                        "razonable dividiendo el problema en partes más manejables y combinando\n" \
                        "las soluciones parciales."
    explicacion = ctk.CTkLabel(app, text=explicacion_texto)
    explicacion.grid(row=1, column=0, columnspan=4, padx=10, pady=20, sticky="ew")

    # Función para calcular la distancia total de una ruta
    def calcularDistancia(ruta, matrizDistancia):
        distanciaTotal = 0
        for i in range(len(ruta) - 1):
            distanciaTotal += matrizDistancia[ruta[i]][ruta[i+1]]
        distanciaTotal += matrizDistancia[ruta[-1]][ruta[0]]
        return distanciaTotal

    # Función para resolver el problema del TSP mediante divide y vencerás
    def divideVencerasVRP(ubicaciones, matrizDistancia, distancias):
        if len(ubicaciones) <= 2:
            distanciaRuta = calcularDistancia(ubicaciones, matrizDistancia)
            distancias.append(distanciaRuta)
            return ubicaciones, distanciaRuta

        mid = len(ubicaciones) // 2
        izquierdaUbicaciones = ubicaciones[:mid]
        derechaUbicaciones = ubicaciones[mid:]

        izquierdaRuta, izquierdaDistancia = divideVencerasVRP(izquierdaUbicaciones, matrizDistancia, distancias)
        derechaRuta, derechaDistancia = divideVencerasVRP(derechaUbicaciones, matrizDistancia, distancias)

        rutasCombinadas = izquierdaRuta + derechaRuta
        distanciaCombinadas = izquierdaDistancia + derechaDistancia + matrizDistancia[izquierdaRuta[-1]][derechaRuta[0]]
        distancias.append(distanciaCombinadas)

        return rutasCombinadas, distanciaCombinadas

    # Función para obtener la matriz de distancias ingresada por el usuario
    def obtenerMatrizDistancia():
        try:
            matriz = []
            for i in range(4):
                fila = []
                for j in range(4):
                    entry = float(entry_matrix[i][j].get())
                    fila.append(entry)
                matriz.append(fila)
            return matriz
        except ValueError:
            messagebox.showerror("Error", "Introduce números válidos, por favor.")
            return None

    # Función para resolver el problema y mostrar los resultados en la interfaz gráfica
    def solucionDivideVenceras():
        matrizDistancia = obtenerMatrizDistancia()
        if matrizDistancia:
            ubicaciones = list(range(len(matrizDistancia)))
            distancias = []
            mejorRuta, minimaDistancia = divideVencerasVRP(ubicaciones, matrizDistancia, distancias)

            mejorRutaText = "Mejor ruta: " + str(mejorRuta)
            mejorRutaTextLabel = ctk.CTkLabel(app, text=mejorRutaText)
            mejorRutaTextLabel.grid(row=6, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

            minimaDistanciaText = "Distancia mínima: " + str(minimaDistancia)
            minimaDistanciaTextLabel = ctk.CTkLabel(app, text=minimaDistanciaText)
            minimaDistanciaTextLabel.grid(row=7, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

            plt.figure(figsize=(10, 5))
            plt.plot(distancias, marker='o')
            plt.title('Distancias de rutas generadas por Divide y Vencerás')
            plt.xlabel('Iteración')
            plt.ylabel('Distancia')
            plt.grid(True)
            plt.show()

    # Creación de la interfaz gráfica para ingresar la matriz de distancias
    matrizDistanciaLabels = []
    entry_matrix = []

    for i in range(4):
        row_labels = []
        row_entries = []
        for j in range(4):
            entry = ctk.CTkEntry(app, width=50)
            entry.grid(row=2+i, column=j, padx=5, pady=5)
            row_entries.append(entry)
        matrizDistanciaLabels.append(row_labels)
        entry_matrix.append(row_entries)

    # Botón para resolver el problema y mostrar los resultados
    resolverBoton = ctk.CTkButton(app, text="Resolver", command=solucionDivideVenceras)
    resolverBoton.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

    # Botón para regresar al menú principal
    regresarBoton = ctk.CTkButton(app, text="Regresar", command=menu)
    regresarBoton.grid(row=9, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

def programacionDinamica():
    limpiarPantalla()   
    dinamicaTitulo = ctk.CTkLabel(app, text="PROBLEMA DEL VIAJANTE DE COMERCIO (TSP)", font=("Helvetica", 20))
    dinamicaTitulo.grid(row=0, column=0, columnspan=4, padx=20, pady=20, sticky="ew")
    explicacion_texto = "La idea es explorar todas las posibles rutas utilizando una máscara\n"\
                        "de bits para representar las ubicaciones visitadas. La función visit\n" \
                        "calcula recursivamente la distancia mínima, almacenando los resultados\n" \
                        "intermedios en una tabla de memoización para evitar cálculos redundantes.\n"
    explicacion = ctk.CTkLabel(app, text=explicacion_texto)
    explicacion.grid(row=1, column=0, columnspan=4, padx=10, pady=20, sticky="ew")

    # Función para resolver el problema del TSP mediante programación dinámica
    def tsp_dynamic_programming(distance_matrix):
        n = len(distance_matrix)
        all_visited = (1 << n) - 1
        memo = [[None] * n for _ in range(1 << n)]

        def visit(mask, pos):
            if mask == all_visited:
                return distance_matrix[pos][0]
            if memo[mask][pos] is not None:
                return memo[mask][pos]

            ans = float('inf')
            for city in range(n):
                if mask & (1 << city) == 0:
                    new_mask = mask | (1 << city)
                    ans = min(ans, distance_matrix[pos][city] + visit(new_mask, city))

            memo[mask][pos] = ans
            return ans

        return visit(1, 0)

    # Función para obtener la matriz de distancias ingresada por el usuario
    def obtenerMatrizDistancia():
        try:
            matriz = []
            for i in range(4):
                fila = []
                for j in range(4):
                    entry = float(entry_matrix[i][j].get())
                    fila.append(entry)
                matriz.append(fila)
            return matriz
        except ValueError:
            messagebox.showerror("Error", "Introduce números válidos, por favor.")
            return None

    def solucionProgramacionDinamica():
        matrizDistancia = obtenerMatrizDistancia()
        if matrizDistancia:
            min_distance = tsp_dynamic_programming(matrizDistancia)

            min_distance_text = "Distancia mínima: %.2f" % min_distance
            min_distance_label = ctk.CTkLabel(app, text=min_distance_text)
            min_distance_label.grid(row=6, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

            # Aquí podrías agregar la lógica para mostrar una gráfica si lo deseas
            # Por ejemplo, podrías crear un gráfico de barras con los resultados de las distancias mínimas
            distancias_minimas = [min_distance] * 10  # Aquí deberías reemplazar con los datos reales
            iteraciones = list(range(1, 11))  # Suponiendo que tienes 10 iteraciones
            plt.figure(figsize=(8, 5))
            plt.bar(iteraciones, distancias_minimas, color='blue')
            plt.title('Distancia mínima en cada iteración')
            plt.xlabel('Iteración')
            plt.ylabel('Distancia mínima')
            plt.grid(True)
            plt.show()
        
    # Creación de la interfaz gráfica para ingresar la matriz de distancias
    matrizDistanciaLabels = []
    entry_matrix = []

    for i in range(4):
        row_labels = []
        row_entries = []
        for j in range(4):
            entry = ctk.CTkEntry(app, width=50)
            entry.grid(row=2+i, column=j, padx=5, pady=5)
            row_entries.append(entry)
        matrizDistanciaLabels.append(row_labels)
        entry_matrix.append(row_entries)

    # Botón para resolver el problema y mostrar los resultados
    resolverBoton = ctk.CTkButton(app, text="Resolver", command=solucionProgramacionDinamica)
    resolverBoton.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

    # Botón para regresar al menú principal
    regresarBoton = ctk.CTkButton(app, text="Regresar", command=menu)
    regresarBoton.grid(row=9, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

def voraz():
    limpiarPantalla()   
    vorazTitulo = ctk.CTkLabel(app, text="PROBLEMA DEL VIAJANTE DE COMERCIO (TSP)", font=("Helvetica", 20))
    vorazTitulo.grid(row=0, column=0, columnspan=4, padx=20, pady=20, sticky="ew")
    explicacion_texto = "El algoritmo comienza en una ubicación inicial y en cada paso selecciona\n"\
                        "la ubicación no visitada más cercana. La selección se basa en la distancia\n" \
                        "mínima desde la ubicación actual a cualquiera de las ubicaciones no visitadas.\n" \
                        "Este proceso se repite hasta que todas las ubicaciones hayan sido visitadas.\n" \
                        "Finalmente, el algoritmo regresa a la ubicación inicial, completando el ciclo.\n"    
    explicacion = ctk.CTkLabel(app, text=explicacion_texto)
    explicacion.grid(row=1, column=0, columnspan=4, padx=10, pady=20, sticky="ew")


    # Función para calcular la distancia entre dos ubicaciones
    def calcularDistancia(ubicacion1, ubicacion2):
        return math.sqrt((ubicacion2[0] - ubicacion1[0])**2 + (ubicacion2[1] - ubicacion1[1])**2)
    
    def obtenerMatrizDistancia():
        try:
            matriz = []
            for i in range(4):
                fila = []
                for j in range(4):
                    entry = float(entry_matrix[i][j].get())
                    fila.append(entry)
                matriz.append(fila)
            return matriz
        except ValueError:
            messagebox.showerror("Error", "Introduce números válidos, por favor.")
            return None

    # Función para encontrar la ruta voraz
    def encontrarRutaVoraz(matrizDistancia):
        numUbicaciones = len(matrizDistancia)
        mejorRuta = [0]  # Comenzamos desde la ubicación 0
        visitados = set([0])  # Marcamos la ubicación 0 como visitada
        distanciaTotal = 0

        # Iteramos hasta visitar todas las ubicaciones
        while len(visitados) < numUbicaciones:
            mejorDistancia = float('inf')
            mejorUbicacion = None

            # Buscamos la ubicación más cercana que aún no ha sido visitada
            for i in range(numUbicaciones):
                if i not in visitados:
                    distancia = matrizDistancia[mejorRuta[-1]][i]
                    if distancia < mejorDistancia:
                        mejorDistancia = distancia
                        mejorUbicacion = i
            
            # Añadimos la mejor ubicación a la ruta y actualizamos la distancia total y el conjunto de visitados
            mejorRuta.append(mejorUbicacion)
            distanciaTotal += mejorDistancia
            visitados.add(mejorUbicacion)

        # Agregamos el retorno a la ubicación inicial
        mejorRuta.append(0)
        distanciaTotal += matrizDistancia[mejorRuta[-2]][0]

        return mejorRuta, distanciaTotal

    # Función para resolver el problema y mostrar los resultados en la interfaz gráfica
    def solucionVoraz():
        matrizDistancia = obtenerMatrizDistancia()
        if matrizDistancia:
            mejorRuta, distanciaTotal = encontrarRutaVoraz(matrizDistancia)

            mejorRutaText = "Mejor ruta: " + str(mejorRuta)
            mejorRutaTextLabel = ctk.CTkLabel(app, text=mejorRutaText)
            mejorRutaTextLabel.grid(row=6, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

            minimaDistanciaText = "Distancia mínima: " + str(distanciaTotal)
            minimaDistanciaTextLabel = ctk.CTkLabel(app, text=minimaDistanciaText)
            minimaDistanciaTextLabel.grid(row=7, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

            # Crear una matriz de distancias para graficar
            distancias = [matrizDistancia[mejorRuta[i]][mejorRuta[i+1]] for i in range(len(mejorRuta) - 1)]

            plt.figure(figsize=(10, 5))
            plt.plot(distancias, marker='o')
            plt.title('Distancias de rutas generadas por el algoritmo voraz')
            plt.xlabel('Iteración')
            plt.ylabel('Distancia')
            plt.grid(True)
            plt.show()

    # Creación de la interfaz gráfica para ingresar la matriz de distancias
    matrizDistanciaLabels = []
    entry_matrix = []

    for i in range(4):
        row_labels = []
        row_entries = []
        for j in range(4):
            entry = ctk.CTkEntry(app, width=50)
            entry.grid(row=2+i, column=j, padx=5, pady=5)
            row_entries.append(entry)
        matrizDistanciaLabels.append(row_labels)
        entry_matrix.append(row_entries)

    # Botón para resolver el problema y mostrar los resultados
    resolverBoton = ctk.CTkButton(app, text="Resolver", command=solucionVoraz)
    resolverBoton.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

    # Botón para regresar al menú principal
    regresarBoton = ctk.CTkButton(app, text="Regresar", command=menu)
    regresarBoton.grid(row=9, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

def algoritmoMeta():
    limpiarPantalla()
    import csv
    filepath = ""
    
    divideTitulo = ctk.CTkLabel(app, text="PROBLEMA DE RUTEO DE VEHÍCULOS (VRP)", font=("Helvetica", 20))
    divideTitulo.grid(row=0, column=0, columnspan=4, padx=20, pady=20, sticky="ew")
    explicacion_texto = "Este VRP se resuelve utilizando un algoritmo genético. El VRP implica encontrar\n"\
                        "la ruta óptima para un conjunto de vehículos que deben entregar mercancías a varios\n" \
                        "destinos (clientes) desde un punto de partida (depósito) y regresar al mismo punto,\n" \
                        "minimizando la distancia total recorrida o algún otro criterio de optimización"
    explicacion = ctk.CTkLabel(app, text=explicacion_texto)
    explicacion.grid(row=1, column=0, columnspan=4, padx=10, pady=20, sticky="ew")

    def run_metaheuristic(vrp_data):
        print("Ejecutando algoritmo metaheurístico con los datos del VRP...")

    def importarCSV():
        nonlocal filepath
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            vrp_data = read_csv(filepath)
            if vrp_data:
                print("Datos del VRP leídos correctamente.")
                run_metaheuristic(vrp_data)
                return filepath

    def read_csv(filepath):
        vrp = {'nodes': []}
        try:
            with open(filepath, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['type'] == 'param' and row['label'] == 'capacity':
                        vrp['capacity'] = float(row['demand'])
                        if vrp['capacity'] <= 0:
                            raise ValueError('Capacity must be greater than zero.')
                    elif row['type'] == 'node':
                        node = {
                            'label': row['label'],
                            'demand': float(row['demand']),
                            'posX': float(row['posX']),
                            'posY': float(row['posY'])
                        }
                        if node['demand'] <= 0:
                            raise ValueError(f"Demand of node {node['label']} must be greater than zero.")
                        if node['demand'] > vrp['capacity']:
                            raise ValueError(f"Demand of node {node['label']} exceeds vehicle capacity.")
                        vrp['nodes'].append(node)
            if 'capacity' not in vrp:
                raise ValueError('Missing capacity parameter.')
            if len(vrp['nodes']) == 0:
                raise ValueError('No nodes found.')
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None
        return vrp

    def distance(n1, n2):
        dx = n2['posX'] - n1['posX']
        dy = n2['posY'] - n1['posY']
        return math.sqrt(dx * dx + dy * dy)

    def fitness(p):
        s = distance(vrp['nodes'][0], vrp['nodes'][p[0]])
        for i in range(len(p) - 1):
            prev = vrp['nodes'][p[i]]
            next = vrp['nodes'][p[i + 1]]
            s += distance(prev, next)
        s += distance(vrp['nodes'][p[len(p) - 1]], vrp['nodes'][0])
        return s

    def adjust(p):
        repeated = True
        while repeated:
            repeated = False
            for i1 in range(len(p)):
                for i2 in range(i1):
                    if p[i1] == p[i2]:
                        haveAll = True
                        for nodeId in range(len(vrp['nodes'])):
                            if nodeId not in p:
                                p[i1] = nodeId
                                haveAll = False
                                break
                        if haveAll:
                            del p[i1]
                        repeated = True
                    if repeated: break
                if repeated: break

        i = 0
        s = 0.0
        cap = vrp['capacity']
        while i < len(p):
            s += vrp['nodes'][p[i]]['demand']
            if s > cap:
                p.insert(i, 0)
                s = 0.0
            i += 1

        i = len(p) - 2
        while i >= 0:
            if p[i] == 0 and p[i + 1] == 0:
                del p[i]
            i -= 1

    filepath = importarCSV()
    vrp = read_csv(filepath)
    if vrp is None:
        print("Failed to read VRP data.")
        exit(1)
    else:
        popsize = 50
        iterations = 100
        pop = []
        for i in range(popsize):
            p = list(range(1, len(vrp['nodes'])))
            random.shuffle(p)
            pop.append(p)
        for p in pop:
            adjust(p)

        for i in range(iterations):
            nextPop = []
            for j in range(int(len(pop) / 2)):
                parentIds = set()
                while len(parentIds) < 4:
                    parentIds |= {random.randint(0, len(pop) - 1)}
                parentIds = list(parentIds)
                parent1 = pop[parentIds[0]] if fitness(pop[parentIds[0]]) < fitness(pop[parentIds[1]]) else pop[parentIds[1]]
                parent2 = pop[parentIds[2]] if fitness(pop[parentIds[2]]) < fitness(pop[parentIds[3]]) else pop[parentIds[3]]

                cutIdx1, cutIdx2 = random.randint(1, min(len(parent1), len(parent2)) - 1), random.randint(1, min(len(parent1), len(parent2)) - 1)
                cutIdx1, cutIdx2 = min(cutIdx1, cutIdx2), max(cutIdx1, cutIdx2)
                child1 = parent1[:cutIdx1] + parent2[cutIdx1:cutIdx2] + parent1[cutIdx2:]
                child2 = parent2[:cutIdx1] + parent1[cutIdx1:cutIdx2] + parent2[cutIdx2:]
                nextPop += [child1, child2]

            if random.randint(1, 15) == 1:
                ptomutate = nextPop[random.randint(0, len(nextPop) - 1)]
                i1 = random.randint(0, len(ptomutate) - 1)
                i2 = random.randint(0, len(ptomutate) - 1)
                ptomutate[i1], ptomutate[i2] = ptomutate[i2], ptomutate[i1]

            for p in nextPop:
                adjust(p)

            pop = nextPop

        better = None
        bf = float('inf')
        for p in pop:
            f = fitness(p)
            if f < bf:
                bf = f
                better = p

        ruta_text = "Ruta: " + ' -> '.join([vrp['nodes'][nodeIdx]['label'] for nodeIdx in better])
        ruta_text_label = ctk.CTkLabel(app, text=ruta_text)
        ruta_text_label.grid(row=7, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        depot_text = "Costo: %.2f" % bf
        depot_text_label = ctk.CTkLabel(app, text=depot_text)
        depot_text_label.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        x_coords = [node['posX'] for node in vrp['nodes']]
        y_coords = [node['posY'] for node in vrp['nodes']]

        # Añadir las coordenadas del depósito al principio de las listas
        x_coords.insert(0, 0)
        y_coords.insert(0, 0)

        # Crear una lista para almacenar el orden de visita de los nodos en la mejor solución
        route_order = [better[0]] + better + [better[0]]  # Añadir el depósito al final

        # Crear listas para almacenar las coordenadas de las rutas
        route_x = [x_coords[node_idx] for node_idx in route_order]
        route_y = [y_coords[node_idx] for node_idx in route_order]

        # Mostrar la gráfica con las rutas y el depósito
        plt.figure(figsize=(8, 6))
        plt.plot(route_x, route_y, marker='o', linestyle='-')
        plt.plot(x_coords[0], y_coords[0], marker='s', color='red', markersize=10)  # Depósito
        plt.title('Rutas Generadas por el Algoritmo VRP')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.grid(True)
        plt.show()

    # Interfaz gráfica para importar el archivo CSV
    importButton = ctk.CTkButton(app, text="Importar CSV", command=importarCSV)
    importButton.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

    # Botón para volver al menú principal
    backButton = ctk.CTkButton(app, text="Volver al menú", command=menu)
    backButton.grid(row=3, column=0, columnspan=4, padx=10, pady=10, sticky="ew")


    
    
app = ctk.CTk()
app.title("VRP")
app.geometry("500x600")
app.resizable(1, 1)

Titulo = ctk.CTkLabel(app, text="PROBLEMA DE RUTEO DE VEHÍCULOS", font=("Helvetica", 20))
Titulo.grid(row=0, column=1, padx=60, pady=40, sticky="nsew")

fuerzaBrutaBoton = ctk.CTkButton(app, text="Fuerza bruta | TSP", command=fuerzaBruta, width=200)
fuerzaBrutaBoton.grid(row=1, column=1, pady=5)

divideVencerasBoton = ctk.CTkButton(app, text="Divide y vencerás | TSP", command=divideVenceras, width=200)
divideVencerasBoton.grid(row=2, column=1, pady=5)

dinamicaBoton = ctk.CTkButton(app, text="Programación dinámica | TSP", command=programacionDinamica, width=200)
dinamicaBoton.grid(row=3, column=1, pady=5)

greedyBoton = ctk.CTkButton(app, text="Voráz | TSP", command=voraz, width=200)
greedyBoton.grid(row=4, column=1, pady=5)

metaBoton = ctk.CTkButton(app, text="Algoritmo metaheurístico", command=algoritmoMeta, width=200)
metaBoton.grid(row=5, column=1, pady=5)

result_label = ctk.CTkLabel(app, text="")
result_label.grid(row=6, column=1, pady=20)

app.mainloop()