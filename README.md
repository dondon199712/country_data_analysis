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
<img width="500" src="https://github.com/dondon199712/country_data_analysis/blob/main/AxesSubplot.png" alt="AxesSubplot:">

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
**Instructions:**

1. Clone this repository to your local machine using Git.
2. Open the `country_analysis.ipynb` file in a Jupyter Notebook environment.
3. Make sure you have the required libraries installed (`pandas`, `scikit-learn`, etc.).
4. Run the notebook cells to reproduce the analysis.



