
import pandas as pd
from sklearn.cluster import KMeans
import h5py

# Load test data from CSV file
test_data = pd.read_csv('data/song data/songs.csv')

# Filter test data by year
#year = input('Enter year to filter test data: ')
#year2=int(year)+5
#emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
filtered_data = test_data[(test_data['year'] >= 2000) & (test_data['year'] <= 2005]


# Get input from user
emotion = input('Enter emotion to print cluster: ')

# Load the model from h5 file
with h5py.File('model/kmeans_model.h5', 'r') as f:
    centroids = f['centroids'][()]
    labels = f['labels'][()]

print('model loaded')


# Get cluster for specified emotion
#filtered_labels = labels[filtered_data['valence'] == int(emotion)]


# Print the cluster
#print(f'Cluster for emotion {emotion}: {filtered_labels}')



#labels = kmeans.predict(X)
cluster_number = int(emotion)
columns = ['artists','name','year']
cluster_data = filtered_data[labels == cluster_number]
print(cluster_data[columns])
