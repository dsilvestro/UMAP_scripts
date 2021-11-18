import umap.plot
import umap.umap_ as umap
import numpy as np
np.set_printoptions(suppress=True,precision=3)  
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
import pandas as pd
from copy import deepcopy

# PREP DATA
tbl = pd.read_csv("Australia_biome_BIOCLIM.txt", delimiter="\t")


biome_names = ['Tropical forest', # 1           "#31a354"
               'Temperate forest', # 4          "#addd8e"
               'Tropical savanna', # 7          "#efa32b"
               'Temperate grassland', # 8       "#fad71d"
               'Montane grassland', # 10        "#c4a172"
               'Mediterranean forest', # 12     "#d9232a"
               'Desert & Xeric shrublands' # 13 "#e0785b"
]


colnames = [i for i in tbl.columns if 'bio' in i]
colnames = colnames[1:] # drop 'biome' col
colnames = colnames + ['lon', 'lat']

tbl_dropna = tbl.dropna()
biome_data = tbl_dropna[colnames].values
biome_numeric = [x for x in tbl_dropna.biome.map({1:0, 4:1, 7:2, 8:3, 10:4, 12:5, 13:6})]
biome_scaler = StandardScaler().fit(biome_data)
scaled_biome_data = StandardScaler().fit_transform(biome_data)

colors = np.array(["#31a354","#addd8e","#efa32b","#fad71d","#c4a172","#d9232a","#e0785b"])
col_list = colors[biome_numeric]




######
# UNSUPERVISED
reducer = umap.UMAP(n_neighbors=250, min_dist=0)
umap_obj = reducer.fit(scaled_biome_data)
embedding = reducer.transform(scaled_biome_data)

# compare with PCA
pca = PCA(n_components=2)
pca.fit(scaled_biome_data)
pca_embed = pca.transform(scaled_biome_data)


# PLOT
plot_scaler = 1.1
alpha=0.7
fig = plt.figure(figsize=(10*plot_scaler, 5*plot_scaler))
ax = fig.add_subplot(121)

cax = ax.scatter(embedding[:, 0],
                 embedding[:, 1],
                 c=col_list, alpha=alpha)
ax.set_title('UMAP', fontsize=24)
ax.set_xlabel('umap #1', fontsize=12)
ax.set_ylabel('umap #2', fontsize=12)


ax = fig.add_subplot(122)
cax = ax.scatter(
    pca_embed[:, 0],
    pca_embed[:, 1],
    c=col_list, alpha=alpha)
ax.set_title('PCA', fontsize=24)
ax.set_xlabel('pc1', fontsize=12)
ax.set_ylabel('pc2', fontsize=12)
file_name = "UMAP_vs_PCA-2D.pdf"
fig.tight_layout()
plot_res = matplotlib.backends.backend_pdf.PdfPages(file_name)
plot_res.savefig(fig)
plot_res.close()





# INVERSE tranform
inv_transf = reducer.inverse_transform(embedding)
inv_transformed_prediction = biome_scaler.inverse_transform(inv_transf)
mre = np.round(np.mean(np.abs(inv_transformed_prediction - biome_data)/np.mean(biome_data), axis=0), 3)

# inv transform pca
inv_transf_pca = pca.inverse_transform(pca_embed)
inv_transformed_prediction_pca = biome_scaler.inverse_transform(inv_transf_pca)
mre_pca = np.round(np.mean(np.abs(inv_transformed_prediction_pca - biome_data)/np.mean(biome_data), axis=0), 3)



# PLOTS
plot_scaler = 1.1
alpha=0.7
f1, f2 = -100, 100
fig = plt.figure(figsize=(10*plot_scaler, 10*plot_scaler))
ax = fig.add_subplot(221)
i, j = 3, 18
cax = ax.scatter(
    biome_data[:,i], inv_transformed_prediction[:,i],
    c=col_list, alpha=alpha)
ax.set_title('BIO%s - mre: %s' % (i, round(mre[i],3)), fontsize=14)
ax.set_xlabel('True value', fontsize=12)
ax.set_ylabel('UMAP inverse transform', fontsize=12)
plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
ax.set_ylim(f1+np.min(biome_data[:,i]), f2+np.max(biome_data[:,i]))
ax.set_xlim(f1+np.min(biome_data[:,i]), f2+np.max(biome_data[:,i]))

ax = fig.add_subplot(222)
cax = ax.scatter(
    biome_data[:,j], inv_transformed_prediction[:,j],
    c=col_list,alpha=alpha)
ax.set_title('BIO%s - mre: %s' % (j, round(mre[j],3)), fontsize=14)
ax.set_xlabel('True value', fontsize=12)
ax.set_ylabel('UMAP inverse transform', fontsize=12)
plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
ax.set_ylim(f1+np.min(biome_data[:,j]), f2+np.max(biome_data[:,j]))
ax.set_xlim(f1+np.min(biome_data[:,j]), f2+np.max(biome_data[:,j]))

ax = fig.add_subplot(223)
cax = ax.scatter(
    biome_data[:,i], inv_transformed_prediction_pca[:,i],
    c=col_list, alpha=alpha)
ax.set_title('BIO%s - mre: %s' % (i, round(mre_pca[i],3)), fontsize=14)
ax.set_xlabel('True value', fontsize=12)
ax.set_ylabel('PCA inverse transform', fontsize=12)
plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
ax.set_ylim(f1+np.min(biome_data[:,i]), f2+np.max(biome_data[:,i]))
ax.set_xlim(f1+np.min(biome_data[:,i]), f2+np.max(biome_data[:,i]))

ax = fig.add_subplot(224)
cax = ax.scatter(
    biome_data[:,j], inv_transformed_prediction_pca[:,j],
    c=col_list,alpha=alpha)
ax.set_title('BIO%s - mre: %s' % (j, round(mre_pca[j],3)), fontsize=14)
ax.set_xlabel('True value', fontsize=12)
ax.set_ylabel('PCA inverse transform', fontsize=12)
plt.axline((0, 0), (1, 1), linewidth=2, linestyle='dashed', alpha=0.5, color="k")
ax.set_ylim(f1+np.min(biome_data[:,j]), f2+np.max(biome_data[:,j]))
ax.set_xlim(f1+np.min(biome_data[:,j]), f2+np.max(biome_data[:,j]))

fig.tight_layout()
file_name = "PCA_2D_inverse_transform.pdf"
plot_res = matplotlib.backends.backend_pdf.PdfPages(file_name)
plot_res.savefig(fig)
plot_res.close()




# SUPERVISED LEARNING
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
col_list = colors[biome_numeric]
reducer = umap.UMAP(target_metric='categorical',n_neighbors=30, min_dist=0.1)
umap_obj = reducer.fit(scaled_biome_data, y=tbl_dropna.biome)
embedding = reducer.transform(scaled_biome_data)
embedding.shape
cax = ax.scatter(
    embedding[:, 0],
    embedding[:, 1], c=col_list)



# SEMISUPERVISED LEARNING
col_list = colors[biome_numeric]
reducer = umap.UMAP(target_metric='categorical',n_neighbors=50,min_dist=0.1)

sparse_labels = deepcopy(tbl_dropna.biome)
indx = np.random.choice(range(len(sparse_labels)), int(0.25 * len(sparse_labels)), replace=False)
indx_rv = np.setdiff1d(np.arange(len(sparse_labels)), indx)                        
sparse_labels.values[indx] = -1
biome_numeric = np.array(biome_numeric)
biome_numeric_sparse = np.array(deepcopy(biome_numeric))
biome_numeric_sparse[indx] = -1

umap_obj = reducer.fit(scaled_biome_data, y=sparse_labels)
embedding = reducer.transform(scaled_biome_data)

ax = fig.add_subplot(122)
cax = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=col_list)
ax.set_title('SSL-UMAP projection of the Australian biomes', fontsize=24)
file_name = "UMAP_semi-supervised0.25.pdf"
fig.tight_layout()
plot_res = matplotlib.backends.backend_pdf.PdfPages(file_name)
plot_res.savefig(fig)
plot_res.close()

