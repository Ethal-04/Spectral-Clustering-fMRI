import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets, plotting, image
from nilearn.input_data import NiftiLabelsMasker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage

fmri_img = nib.load(r'C:\Users\Sheethal\Documents\Datasets\sub-MJF001_task-rest_bold.nii.gz') 

smoothed_img = image.smooth_img(fmri_img, fwhm=9)

aal_atlas = datasets.fetch_atlas_aal()
atlas_img = nib.load(aal_atlas.maps)
atlas_labels = aal_atlas.labels

masker = NiftiLabelsMasker(labels_img=atlas_img,standardize=True,detrend=True)
time_series = masker.fit_transform(smoothed_img)  

plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(time_series[:, i], label=f'Region {i+1}')
plt.title("Time Series of First 10 Brain Regions (AAL Atlas)")
plt.xlabel("Timepoints")
plt.ylabel("Standardized Signal Intensity")
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig("timeseries_10_regions.png")
plt.show()

scaler = StandardScaler()
standardized_ts = scaler.fit_transform(time_series)

corr_matrix = np.corrcoef(standardized_ts.T)

distance_matrix = 1 - corr_matrix

n_clusters = 5
clustering = SpectralClustering(n_clusters=n_clusters,
                                 affinity='precomputed',
                                 assign_labels='kmeans',
                                 random_state=42)
labels = clustering.fit_predict(distance_matrix)

sorted_idx = np.argsort(labels)
from scipy.cluster.hierarchy import leaves_list
linked = linkage(distance_matrix, method='ward')
ordered_idx = leaves_list(linked)
sns.heatmap(distance_matrix[ordered_idx][:, ordered_idx], 
            cmap='viridis', 
            xticklabels=False, 
            yticklabels=False,
            cbar_kws={'label': '1 - Correlation'})
plt.title('Hierarchically Clustered Distance Matrix')

coords = plotting.find_parcellation_cut_coords(labels_img=atlas_img)
coords = np.array(coords)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                     c=labels, cmap='Set1', s=50)
ax.set_title('3D Visualization of Clustered Brain Regions (Spectral Clustering)')
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, label='Cluster Label')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.tight_layout()
plt.savefig("3d_clusters_labeled.png")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
coords_2d = pca.fit_transform(coords)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap='Set1', s=60)
plt.title("2D PCA Projection of Brain Region Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
cbar = plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.tight_layout()
plt.savefig("2d_scatter_clusters_labeled.png")
plt.show()

plt.figure(figsize=(14, 10))
dendrogram(linked, labels=np.arange(116), leaf_rotation=90, color_threshold=50)
plt.title("Dendrogram of Brain Regions (Ward's Method)")
plt.xlabel("Region Index")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.savefig("dendrogram_labeled.png")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar_kws={'label': '1 - Correlation'})
plt.title('Distance Matrix of Brain Regions (1 - Pearson Correlation)')
plt.xlabel('Region Index')
plt.ylabel('Region Index')
plt.tight_layout()
plt.savefig("similarity_matrix.png")
plt.show()