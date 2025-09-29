import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import entropy

# 定义两个矩阵
# matrix1 = np.array([[0, 1],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [1, 0],
#         [0, 1],
#         [1, 0],
#         [1, 0],
#         [0, 1]])

# matrix2 = np.array([[0.0853, 0.9147],
#         [0.8681, 0.1319],
#         [0.7839, 0.2161],
#         [1.0000, 0.0000],
#         [0.7033, 0.2967],
#         [0.0206, 0.9794],
#         [1.0000, 0.0000],
#         [1.0000, 0.0000],
#         [0.0900, 0.9100]])
matrix1 = np.array([[0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0]])

matrix2 = np.array([[0.3651, 0.6349],
        [0.2238, 0.7762],
        [0.0000, 1.0000],
        [0.6226, 0.3774],
        [1.0000, 0.0000],
        [0.8206, 0.1794],
        [0.0000, 1.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000]])

# 欧氏距离
euclidean_distances = [euclidean(matrix1[i], matrix2[i]) for i in range(len(matrix1))]
avg_euclidean_distance1 = np.mean(euclidean_distances)

# 余弦相似度
cosine_similarities = [cosine(matrix1[i], matrix2[i]) for i in range(len(matrix1))]
avg_cosine_similarity1 = np.mean(cosine_similarities)

# KL 散度
kl_divergences = [entropy(matrix1[i], matrix2[i]) for i in range(len(matrix1))]
avg_kl_divergence1 = np.mean(kl_divergences)


# matrix1 = np.array([[0, 1],
#         [1, 0],
#         [0, 1],
#         [1, 0],
#         [1, 0],
#         [0, 1],
#         [1, 0],
#         [0, 1],
#         [1, 0]])

# matrix2 = np.array([[0.1256, 0.8744],
#         [0.5958, 0.4042],
#         [0.0000, 1.0000],
#         [1.0000, 0.0000],
#         [1.0000, 0.0000],
#         [0.0000, 1.0000],
#         [0.8102, 0.1898],
#         [0.2871, 0.7129],
#         [0.9158, 0.0842]])

matrix1 = np.array([[1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]])

matrix2 = np.array([[1.0000, 0.0000],
        [0.0000, 1.0000],
        [0.1619, 0.8381],
        [0.1863, 0.8137],
        [0.3068, 0.6932],
        [0.0000, 1.0000],
        [0.0000, 1.0000],
        [0.2633, 0.7367],
        [0.3990, 0.6010]])
# 欧氏距离
euclidean_distances = [euclidean(matrix1[i], matrix2[i]) for i in range(len(matrix1))]
avg_euclidean_distance2 = np.mean(euclidean_distances)

# 余弦相似度
cosine_similarities = [cosine(matrix1[i], matrix2[i]) for i in range(len(matrix1))]
avg_cosine_similarity2 = np.mean(cosine_similarities)

# KL 散度
kl_divergences = [entropy(matrix1[i], matrix2[i]) for i in range(len(matrix1))]
avg_kl_divergence2 = np.mean(kl_divergences)

print(0.5*(avg_euclidean_distance1+avg_euclidean_distance2))
print(0.5*(avg_cosine_similarity1+avg_cosine_similarity2))
print(0.5*(avg_kl_divergence1+avg_kl_divergence2))
