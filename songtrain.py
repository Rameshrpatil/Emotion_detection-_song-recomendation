import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import h5py
# load data from a CSV file
data = pd.read_csv('data/song data/songs_test.csv')
print('data loaded')


# extract the four features from the dataset
X = data.iloc[:, [16,2,4,9,10,17]].values
#year = 2000#int(input('Enter year to filter test data: '))
#year2=2005#int(year)+5
filtered_data = data[(data['year'] >= 2000) & (data['year'] <= 2005)]

# define the number of clusters to create
k = 7

# create a K-means clustering model with k clusters
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# visualize the clusters on a scatter plot
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Angry')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Disgusted')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Fearful')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Happy')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Neutral')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'yellow', label = 'Sad')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 100, c = 'black', label = 'Surprised')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'gray', label = 'Centroids')
plt.xlabel('liveness')
plt.ylabel('acousticness')
plt.title('K-means Clustering')
plt.legend()
plt.show()


# Save the model as an h5 file
# with h5py.File('kmeans_model.h5', 'w') as f:
#     f.create_dataset('centroids', data=kmeans.cluster_centers_)
#     f.create_dataset('labels', data=kmeans.labels_)
labels = kmeans.predict(X)
cluster_number = 2
columns = ['artists','name',]#'year']
cluster_data = filtered_data[labels == cluster_number]
print(cluster_data[columns])