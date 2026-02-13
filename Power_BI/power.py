import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = sns.load_dataset('iris')
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", ax=axes[0])
sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", ax=axes[1])