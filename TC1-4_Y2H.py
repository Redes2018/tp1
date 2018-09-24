import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')
import numpy as np
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from scipy.optimize import curve_fit
import os

def lineal(x, a, b):
  return a*x + np.log10(b)

# Leemos el archivo
myFolder=(os.getcwd()+'/tc01_data/') #busca en el directorio actual

# Primero creamos la red: 
data1 = pd.read_csv(myFolder+'yeast_Y2H.txt', sep='\t', header=None)
Y2H=nx.Graph()
for i in range(len(data1)):
    Y2H.add_edges_from([(data1[0][i],data1[1][i])])

#eliminamos nodos aislados:
aislados=list(nx.isolates(Y2H))
Y2H.remove_nodes_from(aislados)

#Grafico: Toda la red
fig, ax = plt.subplots()
pos = nx.spring_layout(Y2H)
degrees = Y2H.degree() #Dict with Node ID, Degree
nodos = Y2H.nodes()
n_color = np.asarray([degrees[n] for n in nodos])
nodes=nx.draw_networkx_nodes(Y2H,pos,cmap='tab10',node_color=n_color,node_size=10) 
edges=nx.draw_networkx_edges(Y2H,pos,edge_color='k',width=0.1)
#nx.draw_networkx_labels(Y2H,pos,font_size=10)
fig.colorbar(nodes)
plt.title('Red yeast_Y2H por grados')
plt.show()


#Grafico: componente gigante
giant = max(nx.connected_component_subgraphs(Y2H), key=len)
fig, ax = plt.subplots()
pos = nx.spring_layout(giant)
degrees = giant.degree() #Dict with Node ID, Degree
nodos = giant.nodes()
n_color = np.asarray([degrees[n] for n in nodos])
nodes=nx.draw_networkx_nodes(giant,pos,cmap='tab10',node_color=n_color,node_size=10) 
edges=nx.draw_networkx_edges(giant,pos,edge_color='k',width=0.1)
#nx.draw_networkx_labels(giant,pos,font_size=10)
fig.colorbar(nodes)
plt.title('Red yeast_Y2H por grados')
plt.show()


#Matriz de Adjacencia Aij:
C_adj = nx.to_numpy_matrix(Y2H) 
#N_adj = nx.to_numpy_matrix(net)

nodos = Y2H.number_of_nodes()
#nodosN = net.number_of_nodes()


#i)
# k_nodo es el vector de grado de cada nodo
# knn_nodo es el valor, de grado promedio de vecinos de cada nodo
k_nodo= [val for (node, val) in Y2H.degree()]
knn_nodo=np.zeros(len(k_nodo))

for i in range(len(k_nodo)):
        knn_nodo[i] = np.dot(np.array(C_adj[i,:]), np.array(k_nodo)) / int(k_nodo[i]) #k_nodo[i] numero de vecinos propios para promediar.


#ii)Grafico de knn en funcion del grado k:
#Tenemos que promediar:

#Ordenar el knn_nodo en funcion del key del k_nodo
knn_nodo=[x for _,x in sorted(zip(k_nodo,knn_nodo))]
knn_nodo=np.array(knn_nodo) 
#Ordenamos el k_nodo
k_nodo=np.sort(k_nodo)

#k es el vector de grados
k=np.unique(k_nodo)
knn=np.zeros(len(k))
knn_error=np.zeros(len(k))

for i,j in enumerate(k):
    v=np.where(k_nodo==j)
    knn[i]=np.mean(knn_nodo[v])
    knn_error[i]=np.std(knn_nodo[v],ddof=1)#ddof=significa degree of freedom en el calculo de la desviacion standar.
                                           #Creo que es mejor con ddof=1 porque si es un solo valor(el error no estaria definido)
                                           #y entonces asigna correctamente nan como error en vez de error 0 lo cual no estaria bien.


#iii) Ajuste: con funcion log(knn)=mu*(logk)+log(b)

#AJuste 1: usando los puntos rojos
parametros1=curve_fit(lineal,np.log10(k_nodo),np.log10(knn_nodo))[0]
mu1=parametros1[0]
b1=parametros1[1]
intercept1=np.log10(b1)

print ('Ajuste rojo')
print ('exponente = {}'.format(mu1))
print ('intercept = {}'.format(intercept1))
print('\n')


xfit1=np.linspace(1,50,1000)
yfit1=(b1)*(xfit1**mu1)

#Ajuste 2: usando los puntos azules

parametros2=curve_fit(lineal,np.log10(k),np.log10(knn))[0]
mu2=parametros2[0]
b2=parametros2[1]
intercept2=np.log10(b2)

print ('Ajuste azul')
print ('exponente = {}'.format(mu2))
print ('intercept = {}'.format(intercept2))
print('\n')

xfit2=np.linspace(1,50,1000)
yfit2=(b2)*(xfit2**mu2)

#AJuste 3: usando los puntos azules del grafico knn(k) y usando tambien el error:
#Para el ajuste con errores me quedo con puntos que tengan knn_error != nan
nonan_positions=[i for i,j in enumerate(knn_error) if np.invert(np.isnan(j))==True]
knn_error=knn_error[nonan_positions]
knn_fit=knn[nonan_positions]
k_fit=k[nonan_positions]

parametros3=curve_fit(lineal,np.log10(k_fit),np.log10(knn_fit),sigma=(1/np.log(10))*(1/knn_fit)*knn_error)[0]
mu3=parametros3[0]
b3=parametros3[1]
intercept3=np.log10(b3)

print ('Ajuste puntos azules con errores')
print ('exponente = {}'.format(mu3))
print ('intercept = {}'.format(intercept3))
print('\n')


xfit3=np.linspace(1,50,1000)
yfit3=(b3)*(xfit3**mu3)


#Graficos
plt.figure(3)
#Puntos rojos
plt.plot(k_nodo,knn_nodo,'ro',markersize=1)
#Puntos azules
plt.plot(k,knn,'bo',markersize=3)
#Puntos azules con barra de error=desv standar
plt.errorbar(k_fit,knn_fit,(1/np.log(10))*(1/knn_fit)*knn_error,fmt='bo',ecolor='limegreen',elinewidth=1,capsize=5,markersize=3)

#AJustes:
plt.plot(xfit1,yfit1,color='darksalmon',label='Ajuste puntos rojos')
plt.plot(xfit2,yfit2,color='mediumslateblue',label='Ajuste puntos azules')
plt.plot(xfit3,yfit3,color='limegreen',label='Ajuste puntos azules con errores')
#Propiedades grafico:
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$k$')
plt.ylabel('$knn$')
plt.title('Exponente de Correlacion - Red yeast_Y2H')
plt.legend()
plt.show()

#iv)Asortividad con estimador de Newman:
#Segun libro de Newman se puede calcular r de la siguiente forma:
#r=(S1*Se-S2**2)/(S1*S3-S2**2)

S1=np.sum(k_nodo)
S2=np.sum(k_nodo**2)
S3=np.sum(k_nodo**3)

se=0
enlaces=list(Y2H.edges())
for i,e in enumerate(enlaces):
 se=se+Y2H.degree(enlaces[i][0])*Y2H.degree(enlaces[i][1])
Se=2*se

r=(S1*Se-S2**2)/float(S1*S3-S2**2)

print ('Asortatividad Barabasi (mu) = {}'.format(mu3))
print ('Asortatividad Newman (r) = {}'.format(r))


#Output
'''
punto iii)
Ajuste rojo
exponente = 0.26718253712
intercept = 0.678388242637


Ajuste azul
exponente = -0.163393609351
intercept = 1.16142515614


Ajuste puntos azules con errores
exponente = -0.0129419422802
intercept = 1.09426009106

punto iv)
Asortatividad Barabasi (mu) = -0.0129419422802
Asortatividad Newman (r) = -0.0410672111724
'''
