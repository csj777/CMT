from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np

def format_decimal(num):
    a, b = str(num).split('.')
    return float(a + '.' + b[:2])

def get_jnb_threshold(score_list):
    kclf = KMeans(n_clusters=2)
    data_kmeans = np.array(score_list)
    data_kmeans = data_kmeans.reshape(len(data_kmeans), -1)
    kclf.fit(data_kmeans)
    threshod = kclf.cluster_centers_.reshape(-1)
    res = np.sort(threshod)[::-1]
    res = [format_decimal(r) for r in res]
    return res

def get_gmm_threshold(score_list):
    gmm = GaussianMixture(n_components=2)
    data_gmm = np.array(score_list).reshape(-1, 1)
    gmm.fit(data_gmm)
    gmm_means = np.sort(gmm.means_.flatten())[::-1]
    gmm_means = [format_decimal(r) for r in gmm_means]
    return gmm_means

def get_bmm_threshold(score_list):
    bmm = BayesianGaussianMixture(n_components=2, weight_concentration_prior_type='dirichlet_process')
    data_bmm = np.array(score_list).reshape(-1, 1)
    bmm.fit(data_bmm)
    bmm_means = np.sort(bmm.means_.flatten())[::-1]
    bmm_means = [format_decimal(r) for r in bmm_means]
    return bmm_means

