import math
import random
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

k = 3 #change to 2 for question 1
m = 2 #membership degree

def initialize_membership_matrix(data):
    n = len(data)
    membership_matrix = []
    for i in range(n):
        random_numbers = [random.random() for i in range(k)]
        total = sum(random_numbers)
        temp_list = [x/total for x in random_numbers]
        membership_matrix.append(temp_list)
    membership_matrix = np.asarray(membership_matrix)
    return membership_matrix

def compute_cluster_centers(membership_matrix,data):
    n = len(data)
    # print(membership_matrix)
    # print("\n\n\n\n\n")
    cluster_values = zip(*membership_matrix)
    # print(list(cluster_values))
    # exit()
    cluster_centers = list()
    for j in range(k):
        x = next(cluster_values)
        mem_power = [e ** m for e in x]
        denominator = sum(mem_power)
        temp_num = list()
        for i in range(n):
            data_point = list(data.iloc[i])
            prod = [mem_power[i] * d for d in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [num/denominator for num in numerator]
        cluster_centers.append(center)
    return cluster_centers

def update_membership_value(membership_matrix, cluster_centers, data):
    import operator
    n = len(data)
    for i in range(n):
        x = list(data.iloc[i])
        distances = [np.linalg.norm(list(map(operator.sub,x,cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            numerator_denominator = sum([math.pow(float(distances[j]/distances[c]), float(1/(m-1))) for c in range(k)])
            membership_matrix[i][j] = float(1/numerator_denominator )       
    return membership_matrix

def calculate_clusters(membership_matrix,data):
    n = len(data)
    labels = []
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_matrix[i]))
        labels.append(idx)
    return labels

def jaccard(labels1, labels2):
    n11 = n10 = n01 = 0
    n = len(labels1)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)

if __name__ == "__main__":
    #-------Question 1---------
    # data = [[1,6],[2,5],[3,8],[4,4],[5,7],[6,9]]
    # membership_matrix = [[0.8,0.2],[0.9,0.1],[0.7,0.3],[0.3,0.7],[0.5,0.5],[0.2,0.8]]
    # data = pd.DataFrame(data)
    
    #---------Question 2---------
    data = pd.read_csv("wine.csv")
    labels = data.iloc[:,0].values
    # print(labels.tolist())

    membership_matrix = initialize_membership_matrix(data)
    #----Normalization-----
    membership_matrix = (membership_matrix - membership_matrix.mean(axis=0)) / membership_matrix.std(axis=0)

    MAX_ITER = 100
    for i in range(MAX_ITER):
        cluster_centers = compute_cluster_centers(membership_matrix, data)
        membership_matrix = update_membership_value(membership_matrix, cluster_centers, data)
        cluster_labels = calculate_clusters(membership_matrix, data)
    # print(membership_matrix)
    
    print(cluster_centers)
    cluster_centers = np.asarray(cluster_centers)
    cluster_labels = [i+1 for i in cluster_labels]
    print(cluster_labels)

    from sklearn.metrics import accuracy_score
    print(jaccard(labels, cluster_labels))

    f, axes = plt.subplots(1, 2, figsize=(11,5))
    axes[0].scatter(data.iloc[:,4], data.iloc[:,5], alpha=.9)
    axes[1].scatter(data.iloc[:,4], data.iloc[:,5], c=cluster_labels, alpha=.9)
    axes[1].scatter(cluster_centers[:,4], cluster_centers[:,5], marker="+", s=500, c='black')
    plt.show()
