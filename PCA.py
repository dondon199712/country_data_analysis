import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

# add correct path 
file = '/Country-data.csv'
df = pd.read_csv(file, index_col=0)
df.head()

correlation = df.corr()

plt.figure(figsize=(10, 10))  # Size figures

sns.heatmap(correlation, annot=True, cmap='viridis')

life = df.life_expec
df = df.drop('life_expec', axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # Creation of instance StandardScaler

Z = scaler.fit_transform(df)
from sklearn.decomposition import PCA

pca = PCA()  # Creation of instance PCA

Coord = pca.fit_transform(Z)  # Calculation of the coordinates of the PCA
print('The eigenvalues are :', pca.explained_variance_)

plt.plot(np.arange(1, 9), pca.explained_variance_)

plt.xlabel('Number of factors')

plt.ylabel('Eigenvalues')

print('Ratio :',pca.explained_variance_ratio_)

plt.plot(np.arange(1,9),np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Factor number')

plt.ylabel('Cumsum')

sqrt_eigval = np.sqrt(pca.explained_variance_)

corvar = np.zeros((8, 8))

for k in range(8):

    corvar[:, k] = pca.components_[k, :] * sqrt_eigval[k]

# corvar



fig, axes = plt.subplots(figsize=(9, 9))

axes.set_xlim(-1, 1)

axes.set_ylim(-1, 1)

# display of labels (variable names)

for j in range(8):

    plt.annotate(df.columns[j], (corvar[j, 0], corvar[j, 1]), color='#091158')

    plt.arrow(0, 0, corvar[j, 0]*0.9, corvar[j, 1]*0.9,

              alpha=0.5, head_width=0.03, color='b')



# add axes

plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)

plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)



cercle = plt.Circle((0, 0), 1, color='#16E4CA', fill=False)

axes.add_artist(cercle)

plt.xlabel('AXIS 1')

plt.ylabel('AXIS 2')

plt.show()


q = [0, 0.33, 0.66, 1]



lifex = pd.qcut(life, q)



lifex.value_counts()


#positioning of individuals in the foreground

fig, axes = plt.subplots(figsize=(12,12))

axes.set_xlim(-3,3) #same limits on the x-axis

axes.set_ylim(-3,3) #and on the y-axis

#placement of observation labels

for i in range(127):

    if life[i] in lifex.cat.categories[0]:

        plt.annotate(df.index[i],(Coord[i,0],Coord[i,1]), color='#7FCFF1')

    elif life[i] in lifex.cat.categories[1]:

        plt.annotate(df.index[i],(Coord[i,0],Coord[i,1]), color='#16E4CA')

    else:

        plt.annotate(df.index[i],(Coord[i,0],Coord[i,1]), color='#091158')

            

#add axes

plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)

plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)

plt.xlabel('AXE 1')

plt.ylabel('AXE 2')

#display

plt.show()
