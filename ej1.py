import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os

#----------------------------------------------------------------------
#Ejercicio 1:  Proteinas/Levadura
#----------------------------------------------------------------------

#Leemos el archivo
myFolder=(os.getcwd()+'/tc01_data/') #busca en el directorio actual.
data1 = pd.read_csv(myFolder+'yeast_Y2H.txt', sep='\t', header=None)
data2 = pd.read_csv(myFolder+'yeast_AP-MS.txt', sep='\t', header=None)
data3 = pd.read_csv(myFolder+'yeast_LIT.txt', sep='\t', header=None)


#Para acceder a un valor es data[col][fila]
G1=nx.Graph()

G1.add_nodes_from(data1[0][:])
G1.add_nodes_from(data1[1][:])
nodes1=G1.number_of_nodes()

for i in range(len(data1)):
    G1.add_edges_from([(data1[0][i],data1[1][i])])

edges1=G1.number_of_edges()
#plt.figure()
#plt.subplot(3,1,1)
#nx.draw(G1, width=1, with_labels=False, node_color='blue',node_size=50)



G2=nx.Graph()

G2.add_nodes_from(data2[0][:])
G2.add_nodes_from(data2[1][:])
nodes2=G2.number_of_nodes()

for i in range(len(data2)):
    G2.add_edges_from([(data2[0][i],data2[1][i])])

edges2=G2.number_of_edges()

#plt.subplot(3,1,2)
#nx.draw(G2, width=1, with_labels=False, node_color='blue',node_size=50)



G3=nx.Graph()

G3.add_nodes_from(data3[0][:])
G3.add_nodes_from(data3[1][:])
nodes3=G3.number_of_nodes()

for i in range(len(data3)):
    G3.add_edges_from([(data3[0][i],data3[1][i])])

edges3=G3.number_of_edges()

#plt.subplot(3,1,3)
#nx.draw(G3, width=1, with_labels=False, node_color='blue',node_size=50)
#plt.show()

#El grado medio se puede calcular mediante k_mean=2m/n, donde m es la cantidad de enlaces total y n la cant de nodos
K_mean1=round(2*float(edges1)/nodes1)
K_mean2=round(2*float(edges2)/nodes2)
K_mean3=round(2*float(edges3)/nodes3)

#uso la funcion degree_histogram que en 'y' pone la frecuencia que aparece un cierto grado
a=nx.degree_histogram(G1)
b=nx.degree_histogram(G2)
c=nx.degree_histogram(G3)

#El maximo es el largo que tiene el histograma de grados que hace py
K_max1=len(nx.degree_histogram(G1))
K_max2=len(nx.degree_histogram(G2))
K_max3=len(nx.degree_histogram(G3))

#El minimo es el primer numero no nulo que aparezca en la lista del histograma, para eso uso el iterador next()
#y uso el enumerate() para hacer a la lista a,b,c un iterable
K_min1=next((i for i, x in enumerate(a) if x!=0), None)
K_min2=next((i for i, x in enumerate(b) if x!=0), None)
K_min3=next((i for i, x in enumerate(c) if x!=0), None)

#densidad de la red uso density(G)(d=numero enlaces/enlaces maximos posibles)
d1=nx.density(G1)
d2=nx.density(G2)
d3=nx.density(G3)

#Coef de clustering medio:
#c_1= #triangulos con vertice en 1/triangulos posibles con vertice en 1
#C_mean es el promedio de los c_i sobre todos los nodos de la red.
C_mean1= nx.average_clustering(G1)
C_mean2= nx.average_clustering(G2)
C_mean3= nx.average_clustering(G3)

# Clausura transitiva de la red o Global Clustering o Transitividad:
#C_gclust = 3*nx.triangles(G1)/ sumatoria sobre (todos los posibles triangulos)
C_gclust1=nx.transitivity(G1)
C_gclust2=nx.transitivity(G2)
C_gclust3=nx.transitivity(G3)


#Para ver si estaba direccionada existe 'is.directed()'  

#grado medio = #de p.v / #nodos
#grado max = max # de p.V por c/nodo
#densidad de la red = cantidad de edges en c/nodo / todos los edges entre pares de nodos posibles (hay una funcion que se llama density)

haytabla = pd.DataFrame({"Red":["Y2H","AP-MS","LIT"],
                    "Nodos":[nodes1,nodes2,nodes3],
                     "Enlaces":[edges1,edges2,edges3],
                     "K medio":[K_mean1,K_mean2,K_mean3],
                     "K max":[K_max1,K_max2,K_max3],
                     "K min":[K_min1,K_min2,K_min3],
                     "Dirigida":[False,False,False],
                     "Densidad":[d1,d2,d3],"<C_clust>":[C_mean1,C_mean2,C_mean3],
                     "C_gclust":[C_gclust1,C_gclust2,C_gclust3],
                   })
print (haytabla)











