import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
from scipy import stats
import igraph 
from igraph import statistics


# Ejercicio 3: Red de Mark Newman

#Leemos el archivo
myFolder=(os.getcwd()+'/tc01_data/') #busca en el directorio actual

# Primero creamos la red de sistemas de internet
mysystems=nx.read_gml(myFolder+'as-22july06.gml')

N=mysystems.number_of_nodes()
E=mysystems.number_of_edges()

print 'Numero de nodos {}'.format(N)
print 'Numero de enlaces {}'.format(E)
print 'Dirigida: {}'.format(nx.is_directed(mysystems))

#es no dirigida con lo cual solo hay que ver el numero de vecinos sin preocuparse si son in o out.

#Armamos una lista con el numero de vecinos de cada nodo:
nodos=list(mysystems.nodes)
kgrados=[mysystems.degree(nodo) for nodo in nodos]

#Contamos la cantidad de nodos que tienen un cierto k_grado.
#Usamos la funcion np.unique()
#Guardamos el resultado en la variable histograma:       histograma[0]:grados        histograma[1]:cuentas

histograma=np.unique(kgrados,return_counts=True)
k=histograma[0]
pk=histograma[1]/float(N) #pk=Nk/N donde N es el numero total de nodos

#1)Escala Lineal
plt.figure(1)
plt.plot(k,pk,'bo')
plt.xlabel('$k$')
plt.ylabel('$p_{k}$')
plt.title('Lineal scale')


#2)Bineado Lineal
plt.figure(2)
plt.plot(k,pk,'bo')
plt.xlabel('$log$ $k$')
plt.xscale('log')
plt.ylabel('$log$ $p_{k}$')
plt.yscale('log')
plt.title('Linear Binning')

#3)Cumulative
Pk=[]
for grado in k:
 Pk.append(sum(pk[int(np.where(k==grado)[0]) :]))

plt.figure(3)
plt.plot(k,Pk,'bo')
plt.xlabel('$log$ $k$')
plt.xscale('log')
plt.ylabel('$log$ $P_{k}$')
plt.yscale('log')
plt.title('Cumulative')

 
 
#4)Bineado logaritmico

maxgrado=max(k) #maximo grado

logbin=np.logspace(0,np.log(maxgrado)/np.log(10),num=20,endpoint=True,base=10) #bineado en base 10

histograma_logbin=np.histogram(kgrados,bins=logbin,density=False)#Nota: al no ponerle density=True, no lo normaliza. Lo hacemos asi para hacer nosotros
                                                                 #la cuenta de normalizacion dividiendo por el ancho del bin. Luego comparamos con el histo
                                                                 #normalizado y veamos que los puntos nos caigan bien

#Normalizamos por el ancho de los bines y creamos el vector bin_centros
bin_centros=[]
pk_logbin=[]
for i in range(0,len(logbin)-1):
 bin_centros.append((logbin[i+1]+logbin[i])/2)
 bin_ancho=logbin[i+1]-logbin[i]
 pk_logbin.append(histograma_logbin[0][i]/(bin_ancho*N))#normalizamos por el ancho del bin y por el numero total de nodos

#graficos: histograma normalizado vs nuestra cuenta, para chequear que lo hicimos bien 
plt.figure(4)
plt.plot(bin_centros,pk_logbin,'bo')
plt.hist(kgrados,bins=logbin,density=True,edgecolor='blue',color = "skyblue",alpha=0.2)
plt.xlabel('$log$ $k$')
plt.xscale('log')
plt.ylabel('$log$ $pk$')
plt.yscale('log')
plt.title('Log-binning')



exponente=igraph.statistics.power_law_fit(pk_logbin) #Duda: Aca que vector hay que pasarle, en la documentacion dice que tiene que ser un vector de enteros
                                                     #pero pk_logbin es el de nuestro histograma normalizado que siempre es menor a 1
print exponente

#alpha=-1.155704 #el que nos da la funcion power_law_fit como exponente
alpha=-2         #se me ocurrio probar con un alpha puesto a mano mas cerca de 2, quizas este haciendo cualquiera el power_law_fit...averiguemos bien que vector hay que pasarle...
#siguiendo la nomenclatura de cherno:
ksat= 0.000002   #el que no da la funcino power_law_fit y figura como cutoff
kcut=max(k)      

plt.figure(5)

x=np.linspace(0,max(k),2000)
plt.plot(np.array(bin_centros)+ksat,np.array(pk_logbin)*np.exp(np.array(bin_centros)/kcut),'bo') #de los apuntes del cherno saque esto
plt.plot(x,x**(alpha),'r')
plt.xlabel('$log$ $k$')
plt.xscale('log')
plt.ylabel('$log$ $pk$')
plt.yscale('log')
plt.title('Log-binning')
plt.show()
 







