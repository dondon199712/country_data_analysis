# Country Data Analysis Project

This project analyzes data related to countries, using Python libraries like pandas and scikit-learn. 
The Jupyter Notebook included here  explores the data and performs Principal Component Analysis (PCA) to identify key factors affecting the data. 


  <h2>Read and Load data</h2>

  ```python
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# To avoid warning messages

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

df = pd.read_csv("Country-data.csv", index_col=0)

df.head()

```
<h2>  Display the correlation matrix (heatmap) of the different variables of df </h2>

  ```python
correlation = df.corr()

plt.figure(figsize=(10, 10))  # Figure Size

#From seaborn library sns creates a heatmap visualization of the correlation matrix calculated earlier. 
sns.heatmap(correlation, annot=True, cmap='viridis')


```
<img width="500" src="https://github.com/dondon199712/country_data_analysis/blob/main/AxesSubplot.png" alt="AxesSubplot">

<p>The Correlation Matrix:

The heatmap created (df.corr()) helps visualize the relationships between different variables in df.
Strong positive correlations, like the one between "child_death" and "total_iron," suggest these variables tend to move together (higher child death rates might be linked to lower iron levels).</p>

<h2>PCA identifies the most important directions (axes) of variation in the data, allowing us to focus on the patterns that capture the most information with potentially fewer variables</h2>

```python

#Store the column 'life_expec' in a new object life and to delete it from df.

life = df.life_expec

df = df.drop('life_expec', axis=1)


```

```python

#class to normalize data
from sklearn.preprocessing import StandardScaler

# Creation of instance StandardScaler
scaler = StandardScaler() 

# This part uses the scaler object to normalize the data.
Z = scaler.fit_transform(df)

```
<p>
<li>Import the PCA function from the sklearn.decomposition module.</li>
<li>Create pca, an instance of PCA.</li>
<li> Apply pca to the data and create Coord containing the PCA coordinates using the `fit_transform()` method.
</li> </p>

```python

# Use to perform PCA, a technique for dimensionality reduction.
from sklearn.decomposition import PCA

pca = PCA()  # Creation of instance PCA

Coord = pca.fit_transform(Z)  # Calculation of the coordinates of the PCA

```

<p>The interest of PCA lies in this independence, since the analysis brings out very different types of information and spatial organization for each axis. In addition, the factors are hierarchical and take decreasing shares of the variance, the first axes generally concentrate most of the information, which makes the analysis even easier. We will now look at the share of variance explained for each component:
<li>Display the explained variance for each component using the `explained_variance_` attribute of PCA.</li>
<li>Draw the graph representing the explained variance as a function of the number of components.</li></p>

```python
print('The eigenvalues are :', pca.explained_variance_)

plt.plot(np.arange(1, 9), pca.explained_variance_)

plt.xlabel('Number of factors')

plt.ylabel('Eigenvalues');

```


<img width="500" src="https://github.com/dondon199712/country_data_analysis/blob/main/explained_variance.png" alt = "explained_variance">


<p>The eigenvalues are : [3.48753851 1.47902877 1.15061758 0.93557048 0.65529084 0.15140052
 0.11588049 0.07286556]</p>



```python
#Display the ratio of the explained variance thanks to the attribute explained_variance_ratio for each of the components.
print('Ratio :',pca.explained_variance_ratio_)

#Plot the cumulative sum graph representing the ratio of explained variance versus the number of components.
plt.plot(np.arange(1,9),np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Factor number')

plt.ylabel('Cumsum');

```

<img width="500" src="https://github.com/dondon199712/country_data_analysis/blob/main/explained_variance_ratio.png" alt = "explained_variance_ratio">


<h4> We observe here that for 2 axes, the explained variance is about 62%. 
 We can now draw the correlation circle. It allows us to evaluate the influence of each variable for each axis of representation.
</h4>


```python
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

```

<img width = "500" src="https://github.com/dondon199712/country_data_analysis/blob/main/%20correlation%20circle.png" alt="correlation_circle">


<p>We notice that variables such as 'income', 'gdpp' and 'health' are positively correlated with the first axis, whereas 'child_mort' and 'total_fer' are also positively correlated but negatively. We can then look at the representation of the countries in the two axes chosen by the PCA and observe the influence of the variable 'life_expec' on their representations.</p>

```python
#Transform the Life variable into a 3-class categorical variable named lifex, using the function qcut.
q = [0, 0.33, 0.66, 1]



lifex = pd.qcut(life, q)



lifex.value_counts()

```
<h3>Represent each country on the two axes chosen by the PCA by assigning a color according to the different lifex classes.</h3>

```python
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
```

<img width="500" src="https://github.com/dondon199712/country_data_analysis/blob/main/%20division.png" alt = "division">


The visualization reveals three distinct groups of countries. This suggests that life expectancy plays a role in how countries are positioned on the PCA axes. Countries with higher life expectancy tend to be located in the lower right corner of the graph. This reinforces the idea that the chosen axes effectively represent the relationships between the countries.

**Instructions:**

1. Clone this repository to your local machine using Git.
2. Open the `country_analysis.ipynb` file in a Jupyter Notebook environment.
3. Make sure you have the required libraries installed (`pandas`, `scikit-learn`, etc.).
4. Run the notebook cells to reproduce the analysis.



