import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
from torch.nn import functional as F
import torch
import pdb

def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)

    phi[phi_ < 0] = 2*np.pi - phi[phi_ < 0]
    phi[phi == 2*np.pi] = 0

    return theta, phi

def beam_label(theta, beam):
    estimator=KMeans(n_clusters=beam)
    res=estimator.fit_predict(theta.reshape(-1, 1))
    label=estimator.labels_
    centroids=estimator.cluster_centers_
    return label, centroids[:,0]

# def beam_label(theta, beam):
#     import time
#     start_time = time.time()  # 开始计时
#     print("theta.shape",theta.shape)
#     estimator = KMeans(n_clusters=beam)
#     res = estimator.fit_predict(theta.reshape(-1, 1))

#     fit_predict_time = time.time()  # 记录 fit_predict 完成时间
#     print("Time for fit_predict: {:.6f} seconds".format(fit_predict_time - start_time))

#     label = estimator.labels_
#     centroids = estimator.cluster_centers_

#     end_time = time.time()  # 结束计时
#     print("Time for labels and centroids extraction: {:.6f} seconds".format(end_time - fit_predict_time))

#     print("Total time for beam_label function: {:.6f} seconds".format(end_time - start_time))

#     return label, centroids[:, 0]

def generate_mask(phi, beam, label, idxs, beam_ratio, bin_ratio):
    mask = np.zeros((phi.shape[0])).astype(np.bool)

    for i in range(0, beam, beam_ratio):
        phi_i = phi[label == idxs[i]]
        idxs_phi = np.argsort(phi_i)
        mask_i = (label == idxs[i])
        mask_temp = np.zeros((phi_i.shape[0])).astype(np.bool)
        mask_temp[idxs_phi[::bin_ratio]] = True
        mask[mask_i] = mask_temp

    return mask