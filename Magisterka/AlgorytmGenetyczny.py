import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report


def parametryTopologii():
    '''
	Funkcja zwracająca optymalizowane parametry topologii sieci
    
	Zwraca:
	parametry (dict) - słownik z listą parametrów 
	
    '''
    
    fa = [1, 2, 3] #funkcja aktywacji
    lt = [0, 1] # współczynnik uczenia
    lr = list(range(1, 1001, 1)) # startowy współczynnik uczenia
    hl = list(range(1, 6)) # liczba warstw ukrytych
    nn = list(range(1, 129)) # liczba neuronów w warstwie
    parametry = {0:fa, 1:lt, 2:lr, 3:hl, 4:nn}

    return parametry


def startowaPopulacja(liczebnosc_pop):
    '''
    Funkcja tworząca startową populacje
    
	Parametry:
	liczebnosc_pop (int) - liczba określająca liczebność populacji czyli liczbę chromosomów  
	
    Zwraca:
    populacja (dict) - słownik z listami chromosomow
    
    '''
    populacja = {}
    #liczebnosc_pop = 10

    for j in range(liczebnosc_pop):
        ch2 = np.ones(5, dtype=int)
        for i in range(5):
            ch2[i] = np.random.choice(parametryTopologii()[i])
        populacja[j] = ch2
    return populacja


def mutacja(chromosom):
    '''
    
    Funkcja mutacji - zmieniajaca losowo jeden z parametrow chromosomu
    Wybiera losowy gen i zastepuje jego wartość dowolną wartość dozwoloną dla tego genu
    
    Parametry:
    - chromosom (list) - wektor reprezentujacy chromosom
    
    Zwraca:
    - m_chromosom (list) - lista reprezentująca zmutowany chromosom
    
    '''
    
    # parametr który ma zostać zmutowany
    p = np.random.randint(0, len(chromosom))
    m_chromosom = chromosom.copy()
    pT = parametryTopologii()[p]
    pT.pop(pT.index(m_chromosom[p]))
    m_chromosom[p] = np.random.choice(pT)
	
    return m_chromosom
    

def selekcja(populacja, przystosowanie):
    '''
    Funkcja selekcji - wybierająca chromosomy do krzyżowania
    
    Parametry:
    - populacja (dict) - słownik z aktualną populacją
    - przystosowanie (dict) - słownik z wartościami funkcji przystosowania dla każdego chromosomu w populacji
    
    Zwraca:
    - nowa_populacja (dict) - słownik z nową populacją
    
    '''
    fitness = przystosowanie
    mmsc = MinMaxScaler(feature_range=(0,100)).fit_transform(fitness.reshape(-1, 1))
    s = sum(mmsc)
    mmsc = mmsc/s*100
    f_total = []
    for m, i in enumerate(mmsc):
        for j in range(int(i)):
            f_total.append(m)
			
    parents_pool = np.copy(f_total)
	
    for j in range(len(populacja)):
        
		# wybieranie losowych rodziców metodą ruletki
        father = np.random.choice(parents_pool)
        
		# drugi rodzic nie może być taki sam jak pierwszy
        mother = parents_pool[parents_pool!=father]
        
		# jeśli pozostał tylko jeden rodzic w populacji to muszę go wybrać
        if len(mother) == 0:
            mother = father
        else:
            mother = np.random.choice(mother)
        
		father = populacja[father]
        mother = populacja[mother]
        c = np.random.randint(1, len(father))
        child = np.concatenate([father[:c], mother[c:]])
		# mutacja
        mutationPr = 0.25
        wsp_mutacji = mutationPr/np.exp(i/10) 
        m = np.random.random()
        if m < wsp_mutacji:
            child = mutacja(child)
        nowa_populacja[j] = child
    
    return nowa_populacja

def siec(chromosom):
    '''
    
    Funkcja tworząca nową sieć na podstawie otrzymanego chromosomu
    Parametry:
     - chromosom (list) - lista parametrów topologii
     
    Zwraca:
     - macierz (ndarray) - macierz błędu 
    
    '''
    f_aktywacji = {1:'logistic', 2:'tanh', 3:'relu'}
    wsp_uczenia = {0:'constant', 1:'invscaling'}
    warstwy = tuple(np.ones(chromosom[3], dtype=int)*chromosom[4])
    mlp = MLPClassifier(hidden_layer_sizes=warstwy, activation=f_aktywacji[chromosom[0]],\
                        learning_rate=wsp_uczenia[chromosom[1]], learning_rate_init = 1/(2*chromosom[2]))
    
    mlp.fit(X_train_sc, Y_train)
    prognoza = mlp.predict(X_test_sc)
    macierz = confusion_matrix(Y_test, prognoza)
	
    return macierz
    
	
# macierz kosztu = [TP, FP, TN, FN]
def funkcjaZysku(k):
    '''
    Funkcja obliczajaca wartosci zysku danej sieci na podstawie macierzy kontyngencji składającej się z:
    TruePositive, FalsePositive, TrueNegative, FalseNegative
    
    Parametry:
    k (list) - lista zawierajaca liczbe wartosci każdej z czterech możliwych kombinacji klasyfikacji 
    
    Zwraca:
    koszt (float) - liczba rzeczywista określająca zysk związany z danym efektem klasyfikacji
    
    '''
    
    cost_matrix = np.array([19.9, -0.1, 0, -20.1])
    zysk = np.round(np.dot(k, cost_matrix), 5) 

    return zysk


# Główna część wykonująca 

# liczba iteracji algorytmu 
epochs = 10

# liczba chromosomów w populacji - liczba różnych topologii 
liczebnosc_pop = 10

# wartości funkcji przystosowania dla poszczególnych iteracji
f_fit = {}

average_fitness = np.zeros(liczebnosc_pop)
maximum_fitness = np.zeros(liczebnosc_pop)
minimum_fitness = np.zeros(liczebnosc_pop)

# zbior populacji
zb_pop = {}

populacja = startowaPopulacja(liczebnosc_pop)
for i in range(epochs):
    zb_pop[i] = populacja
    f_fit[i] = np.zeros(liczebnosc_pop)
    fitness = np.zeros(len(populacja))
    for p in range(len(populacja)):
        y = siec(populacja[p])
        # wektor w przechowuje liczby: TP, FP, TN, FN
        w = np.array([y[0][0], y[0][1], y[1][1], y[1][0]])
        fitness[p] = funkcjaZysku(w)
    f_fit[i] = fitness
    average_fitness[i] = np.mean(fitness)
    maximum_fitness[i] = np.max(fitness)
    minimum_fitness[i] = np.min(fitness)
    if minimum_fitness[i] == maximum_fitness[i]:
        print('Najmniejsza i największa wartość funkcji przystosowania są takie same.')
        print('Osiągnięto minimum lokalne.')
        print('Iteracja nr {}'.format(i+1))
        break
    nowa_populacja = selekcja(populacja, fitness)
    populacja = nowa_populacja

