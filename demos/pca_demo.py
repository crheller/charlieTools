import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

u = [2, 2]
cov = np.array([[1, 0.3], [0.3, 1]])
X = np.random.multivariate_normal(u, cov, 200)

# perform pca 
X_pca = X - X.mean(axis=0)

# with numpy
cov = np.cov(X_pca.T)
ev, eig = np.linalg.eig(cov)
eig = eig[:, np.argsort(ev)[::-1]]
X_trans_np = np.matmul(X_pca, eig)

# with sklearn
pca = PCA()
pca.fit(X_pca)

X_trans_sklearn = np.matmul(X_pca, pca.components_)


f, ax = plt.subplots(2, 2)

ax[0, 0].set_title('raw data')
ax[0, 0].plot(X[:, 0], X[:, 1], '.')
ax[0, 0].set_xlabel('x1')
ax[0, 0].set_ylabel('x2')
ax[0, 0].axis('equal')

ax[0, 1].set_title('centered data')
ax[0, 1].plot(X_pca[:, 0], X_pca[:, 1], '.')
ax[0, 1].set_xlabel('x1')
ax[0, 1].set_ylabel('x2')
ax[0, 1].axis('equal')

ax[1, 0].set_title('numpy transformation')
ax[1, 0].plot(X_trans_np[:, 0], X_trans_np[:, 1], '.')
ax[1, 0].set_xlabel('pc1')
ax[1, 0].set_ylabel('pc2')
ax[1, 0].axis('equal')

ax[1, 1].set_title('sklearn transformation')
ax[1, 1].plot(X_trans_sklearn[:, 0], X_trans_sklearn[:, 1], '.')
ax[1, 1].set_xlabel('pc2')
ax[1, 1].set_ylabel('pc1')
ax[1, 1].axis('equal')

f.tight_layout()

plt.show()
