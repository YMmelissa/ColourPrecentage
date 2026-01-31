#import torch
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from testing import GeneralDisplay

def find_clusters(data, n_clusters=3, iter=100):
    centers, labels = initialize_model(data, n_clusters)
    for i in range(iter):
        labels, centers = train_model(data, n_clusters, labels, centers)
    return labels, centers

def train_model(data, n_clusters, labels, centers):
    new_centers = update_centers(data, n_clusters, labels)
    new_label = update_labels(data, new_centers)
    return new_label, new_centers

def update_centers(data, n_clusters, labels):
    new_centers = np.zeros((n_clusters, 3))
    counts = np.zeros(n_clusters)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            label = labels[i][j]
            new_centers[label] += data[i][j]
            counts[label] += 1
    for k in range(n_clusters):
        if counts[k] > 0:
            new_centers[k] /= counts[k]
    return new_centers

def update_labels(data, centers):
    new_labels = np.zeros((data.shape[0], data.shape[1]), dtype=int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            distances = np.linalg.norm(centers - data[i][j], axis=1)
            new_labels[i][j] = np.argmin(distances)
    return new_labels

def compress_data(data,res):
    if res == "High":
        scale = 2
    elif res == "Medium":
        scale = 5
    elif res == "Low":
        scale = 10
    compressed = np.zeros((data.shape[0]//scale,data.shape[1]//scale,3))
    for i in range(data.shape[0]//scale):
        for j in range(data.shape[1]//scale):
            i_end = (i+1)*scale
            if i_end > data.shape[0]:
                i_end = data.shape[0]
            j_end = (j+1)*scale
            if j_end > data.shape[1]:
                j_end = data.shape[1]
            block = data[i*scale:i_end,j*scale:j_end]
            compressed[i][j] = np.mean(block, axis=(0,1))/255.0
    return compressed

def initialize_model(data, n_clusters):
    centers = np.random.rand(n_clusters, 3)
    labels = np.random.randint(n_clusters, size=(data.shape[0], data.shape[1]))
    return centers, labels



def extract_data(image_path):
    image = np.array(Image.open(image_path))
    return image

def find_precentage(labels, target_cluster):
    total_pixels = labels.size
    target_pixels = np.sum(labels == target_cluster)
    percentage = (target_pixels / total_pixels) * 100
    return percentage

if __name__ == "__main__":
    image_path = "tester\\d278f7240915025eb53bad73cca0f3fb.jpg"
    data = extract_data(image_path)
    print(data.shape)
    GeneralDisplay(data)
    compressed = compress_data(data,"Low")
    GeneralDisplay(compressed)
    print(compressed.shape)
    labels, centers = find_clusters(compressed, n_clusters=4, iter=50)
    plt.figure(4)
    for i in range(centers.shape[0]):
        precentage = find_precentage(labels, i)
        print(f"Colour {centers[i]*255}: {precentage:.2f}%")
        plt.subplot(1, centers.shape[0], i+1)
        plt.imshow(np.ones((50,50,3))*centers[i].reshape(1,1,3))
    plt.show()
    
