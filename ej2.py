import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os

#----------------------------------------------------------------------
#Ejercicio 2: Delfines
#----------------------------------------------------------------------

#Leemos el archivo
myFolder=(os.getcwd()+'/tc01_data/') #busca en el directorio actual.

mydolphins = nx.read_gml(myFolder+'new_dolphins.gml')
Gender = pd.read_csv(myFolder+'dolphinsGender.txt', sep='\t', header=None)
delfines = Gender[0]
genero = Gender[1]

for d,g in zip(delfines,genero):
    mydolphins.add_node(d, gender=g)
    

#print dict(mydolphins.nodes.data()) Para ver el grafo como diccionario
#print mydolphins.nodes['Jet']['gender'] Para ver la prop genero en el delfin 'Jet'


#a) Tipos de Layout
layouts=['circular_layout','fruchterman_reingold_layout','kamada_kawai_layout','shell_layout','spectral_layout','spring_layout']

for f,lay in enumerate(layouts): 
 plt.figure(f)   
 nx.draw_networkx(mydolphins,eval('nx.'+lay)(mydolphins),
        width=1,
        node_color=["blue" if g=="m" else "red" if g=="f" else "green" for g in nx.get_node_attributes(mydolphins, "gender").values()], 
        node_size=20,
        with_labels=False
       )
 plt.title(lay)
plt.show()
