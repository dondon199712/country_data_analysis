# Country Data Analysis Project

This project analyzes data related to countries, using Python libraries like pandas and scikit-learn. 
The Jupyter Notebook included here  explores the data and performs Principal Component Analysis (PCA) to identify key factors affecting the data. 


  <li><h2>Read and Load data</h2></li>

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
<li><h2>  Display the correlation matrix (heatmap) of the different variables of df </h2></li>

  ```python
correlation = df.corr()

plt.figure(figsize=(10, 10))  # Figure Size

#From seaborn library sns creates a heatmap visualization of the correlation matrix calculated earlier. 
sns.heatmap(correlation, annot=True, cmap='viridis')


```
<img width="500" src="https://github.com/dondon199712/SQL-project/blob/main/User%20Churn/database.png" alt="AxesSubplot:">




**Instructions:**

1. Clone this repository to your local machine using Git.
2. Open the `country_analysis.ipynb` file in a Jupyter Notebook environment.
3. Make sure you have the required libraries installed (`pandas`, `scikit-learn`, etc.).
4. Run the notebook cells to reproduce the analysis.



