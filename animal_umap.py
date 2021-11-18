import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import umap
import pandas as pd
import requests
import json
from umap import UMAP
import matplotlib.backends.backend_pdf

url = "https://raw.githubusercontent.com/duhaime/umap-zoo/03819ed0954b524919671a72f61a56032099ba11/data/json/"
url = url + "armadillo_2.json"
animal = np.array(json.loads(requests.get(url).text)['3d'])
np.shape(animal)

fig, ax = plt.subplots()
ax.scatter(animal[:,2], animal[:,1], s = 1, c = animal[:,0], alpha = 0.1)
ax.axis('equal')

X_train = animal
Y_train = animal[:, 2]
X_train_flat = X_train

embedder = UMAP(n_neighbors=150, verbose=True)
z_umap = embedder.fit_transform(animal)



# inverse transform 
inv_transf = embedder.inverse_transform(z_umap)

# PLOT
plot_scaler=1.1
fig = plt.figure(figsize=(15*plot_scaler, 5*plot_scaler))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(animal[:,0], animal[:,2], animal[:,1], s = 1, c = animal[:,0], alpha = 0.1)
ax.azim = 5
ax.dist = 7
ax.elev = 5

ax = fig.add_subplot(132)
sc = ax.scatter(
    z_umap[:, 0],
    z_umap[:, 1],
    c=animal[:,0],
    alpha=0.5,
    s=0.1
)
ax.axis('equal')
ax.set_title("UMAP embedding", fontsize=20)

ax = fig.add_subplot(133, projection='3d')
ax.scatter(inv_transf[:,0], inv_transf[:,2], inv_transf[:,1], s = 1, c = animal[:,0], alpha = 0.1)
ax.azim = 5
ax.dist = 7
ax.elev = 5

fig.tight_layout()
file_name = "UMAP_animal_%s.pdf" % animal_id
plot_res = matplotlib.backends.backend_pdf.PdfPages(file_name)
plot_res.savefig(fig)
plot_res.close()

