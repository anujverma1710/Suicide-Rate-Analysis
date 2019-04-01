import pandas as pd
from flask import Flask,request,jsonify
from flask import render_template

import numpy as np
import sys
import random

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn import manifold,preprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

app = Flask(__name__)

attributes = ['year', 'suicides_no', 'population', 'suicides_100k_pop', 'HDI_for_year', 'gdp_for_year', 'gdp_per_capita']

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/display_plots")
def display_plots():
    plotter = request.args.get('plotter', 'pca', type=str)
    sampler = request.args.get('sampler', 'rand', type=str)
    scree = int(request.args.get('scree', '0', type=str))
    highest = int(request.args.get('highest', '0', type=str))
    global dataForSampling
    dataForSampling = pd.DataFrame()

    if bool(scree):
        if bool(highest):
            dataForSampling = scree_highest_pca()
        else:
            dataForSampling = scree_original()
    else:
        if sampler == 'rand':
            if plotter == 'pca':
                dataForSampling = pca_random()
            elif plotter == 'mds_cor':
                dataForSampling = mds_correlation_random()
            elif plotter == 'mds_euc':
                dataForSampling = mds_euclidean_random()
            elif plotter == 'scree':
                dataForSampling = scree_random()
            else:
                dataForSampling = scatter_matrix_random()
        else:
            if plotter == 'pca':
                dataForSampling = pca_adaptive()
            elif plotter == 'mds_cor':
                dataForSampling = mds_correlation_adaptive()
            elif plotter == 'mds_euc':
                dataForSampling = mds_euclidean_adaptive()
            elif plotter == 'scree':
                dataForSampling = scree_adaptive()
            else:
                dataForSampling = scatter_matrix_adaptive()

    return pd.io.json.dumps(dataForSampling)


def plot_kmeans_elbow():
    features = df[attributes]

    data_normalized = preprocessing.normalize(features, norm='l2')
    Ks = range(1, 13)
    km = [KMeans(n_clusters=i).fit(data_normalized) for i in Ks]
    centroids = [X.cluster_centers_ for X in km]
    k_euclid = [cdist(data_normalized, cent, 'euclidean') for cent in centroids]
    dist = [np.min(D, axis=1) for D in k_euclid]
    avg_within = [np.sum(dist) / features.shape[0] for dist in dist]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Ks, avg_within, 'g*-')
    ax.plot(Ks[2], avg_within[2], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r',
            markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow plot of KMeans clustering')
    plt.show()

def clustering():
    plot_kmeans_elbow()
    features = csvData[attributes]
    k=3                     #found by plotting k-elbow
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    labels = kmeans.labels_
    csvData['kMeansCluster'] = pd.Series(labels)
    return csvData['kMeansCluster']


def scree_highest_pca():
    return highestPCA


def scree_original():
    [eigenValues, eigenVectors] = generate_eigenValues(np.array(df.fillna(0)))
    return eigenValues

def random_sampling():
    ranSamples = np.array(df.sample(sampleSize))
    return ranSamples

def adaptive_sampling():
    kMeansCluster0 = csvData[csvData['kMeansCluster'] == 0]
    kMeansCluster1 = csvData[csvData['kMeansCluster'] == 1]
    kMeansCluster2 = csvData[csvData['kMeansCluster'] == 2]

    size_kMeansCluster0 = len(kMeansCluster0) * sampleSize / len(csvData)
    size_kMeansCluster1 = len(kMeansCluster1) * sampleSize / len(csvData)
    size_kMeansCluster2 = len(kMeansCluster2) * sampleSize / len(csvData)

    sample_cluster0 = kMeansCluster0.sample(int(size_kMeansCluster0))
    sample_cluster1 = kMeansCluster1.sample(int(size_kMeansCluster1))
    sample_cluster2 = kMeansCluster2.sample(int(size_kMeansCluster2))

    adpSamples = pd.concat([sample_cluster0, sample_cluster1, sample_cluster2])
    return adpSamples

def three_highest_pca_loadings(data):
    [eigenValues, eigenVectors] = generate_eigenValues(data)
    squaredLoadings = []
    for ftrId in range(0, len(eigenVectors)):
        loadings = 0
        for compId in range(0, 3):
            loadings = loadings + (eigenVectors[compId][ftrId])**2
        squaredLoadings.append(loadings)
    squaredLoadings = sorted(range(len(squaredLoadings)), key=lambda k: squaredLoadings[k], reverse=True)
    return squaredLoadings

def generate_eigenValues(data):
    cov_mat = np.cov(data.T)
    eig_values, eig_vectors = np.linalg.eig(cov_mat)
    return eig_values, eig_vectors

def scree_adaptive():
    [eigenValues, eigenVectors] = generate_eigenValues(adpSamples[attributes])
    return eigenValues

def scree_random():
    [eigenValues, eigenVectors] = generate_eigenValues(ranSamples)
    return eigenValues

def pca_random():
    pca_data = PCA(n_components=2)
    X = ranSamples
    pca_data.fit(X)
    X = pca_data.transform(X)
    dataColumns = pd.DataFrame(X)
    for i in range(0, 2):
        dataColumns[attributes[highestPCA[i]]] = df[attributes[highestPCA[i]]][:sampleSize]
    dataColumns['clusterid'] = csvData['kMeansCluster'][:sampleSize]
    return dataColumns

def pca_adaptive():
    X = adpSamples[attributes]
    pca_data = PCA(n_components=2)
    pca_data.fit(X)
    X = pca_data.transform(X)
    dataColumns = pd.DataFrame(X)
    for i in range(0, 2):
        dataColumns[attributes[highestPCA[i]]] = df[attributes[highestPCA[i]]][:sampleSize]
    dataColumns['clusterid'] = np.nan
    x = 0
    for index, row in adpSamples.iterrows():
        dataColumns['clusterid'][x] = row['kMeansCluster']
        x = x + 1
    return dataColumns


def mds_euclidean_random():
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(ranSamples, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    dataColumns = pd.DataFrame(X)
    for i in range(0, 3):
        dataColumns[attributes[highestPCA[i]]] = df[attributes[highestPCA[i]]][:sampleSize]
    dataColumns['clusterid'] = csvData['kMeansCluster'][:sampleSize]
    return dataColumns

def mds_euclidean_adaptive():
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    X = adpSamples[attributes]
    similarity = pairwise_distances(X, metric='euclidean')
    X = mds_data.fit_transform(similarity)
    dataColumns = pd.DataFrame(X)
    for i in range(0, 3):
        dataColumns[attributes[highestPCA[i]]] = df[attributes[highestPCA[i]]][:sampleSize]

    dataColumns['clusterid'] = np.nan
    x = 0
    for index, row in adpSamples.iterrows():
        dataColumns['clusterid'][x] = row['kMeansCluster']
        x = x + 1
    return dataColumns

def mds_correlation_random():
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(ranSamples, metric='correlation')
    X = mds_data.fit_transform(similarity)
    dataColumns = pd.DataFrame(X)
    for i in range(0, 2):
        dataColumns[attributes[highestPCA[i]]] = df[attributes[highestPCA[i]]][:sampleSize]
    dataColumns['clusterid'] = csvData['kMeansCluster'][:sampleSize]
    return dataColumns

def mds_correlation_adaptive():
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    X = adpSamples[attributes]
    similarity = pairwise_distances(X, metric='correlation')
    X = mds_data.fit_transform(similarity)
    dataColumns = pd.DataFrame(X)
    for i in range(0, 2):
        dataColumns[attributes[highestPCA[i]]] = df[attributes[highestPCA[i]]][:sampleSize]

    dataColumns['clusterid'] = np.nan
    x = 0
    for index, row in adpSamples.iterrows():
        dataColumns['clusterid'][x] = row['kMeansCluster']
        x = x + 1
    return dataColumns

def scatter_matrix_random():
    dataColumns = pd.DataFrame()
    for i in range(0, 3):
        dataColumns[attributes[highestPCA[i]]] = df[attributes[highestPCA[i]]][:sampleSize]
        dataColumns['clusterid'] = csvData['kMeansCluster'][:sampleSize]
    return dataColumns

def scatter_matrix_adaptive():
    dataColumns = pd.DataFrame()
    for i in range(0, 3):
        dataColumns[attributes[highestPCA[i]]] = adpSamples[attributes[highestPCA[i]]][:sampleSize]
    dataColumns['clusterid'] = np.nan
    for index, row in adpSamples.iterrows():
        dataColumns['clusterid'][index] = row['kMeansCluster']
    dataColumns = dataColumns.reset_index(drop=True)
    return dataColumns


if __name__ == "__main__":
    global df
    global csvData
    global adpSamples
    global ranSamples
    global highestPCA
    global sampleSize
    df = pd.read_csv("SuicideData.csv",usecols=attributes)
    df = df.fillna(0)
    csvData = df
    scaler = StandardScaler()
    csvData[attributes] = scaler.fit_transform(csvData[attributes])

    sampleSize = 500
    highestPCA = []
    adpSamples = []

    csvData['kMeansCluster'] = clustering()
    adpSamples = adaptive_sampling()
    ranSamples = random_sampling()
    highestPCA = three_highest_pca_loadings(csvData[attributes])
    app.run(host='127.0.0.1',port=5000,debug=True)
