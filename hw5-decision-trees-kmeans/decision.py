import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

def toyProblem():
  X = np.random.randn(150,2)*1.5
  X[:50,:]   += np.array([1,4])
  X[50:100,:]  += np.array([15,-2])
  X[100:,:]  += np.array([5,-2])

  np.random.seed()

  k = 3
  max_iters = 20
  centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False)
  plotClustering(centroids, assignments, X, title="Final Clustering")
  plt.savefig("q4_final_clustering.png", dpi=100)
  plt.close()

  plt.figure(figsize=(16,8))
  plt.plot(SSE, marker='o')
  plt.xlabel("Iteration")
  plt.ylabel("SSE")
  plt.text(k/2, (max(SSE)-min(SSE))*0.9+min(SSE), "k = "+str(k))
  plt.savefig("q4_sse_iters.png", dpi=100)
  plt.close()

  #############################
  # Q5 Randomness in Clustering
  #############################
  k = 5
  max_iters = 20

  SSE_rand = []
  for _ in range(50):
    _, _, sse_hist = kMeansClustering(X, k=k, max_iters=max_iters)
    SSE_rand.append(sse_hist[-1])

  plt.figure(figsize=(8,8))
  plt.hist(SSE_rand, bins=20)
  plt.xlabel("SSE")
  plt.ylabel("# Runs")
  plt.title("Q5 – SSE Distribution (50 random inits, k=5)")
  plt.savefig("q5_sse_distribution.png", dpi=100)
  plt.close()
  print(f"Q5 SSE min={min(SSE_rand):.2f}  max={max(SSE_rand):.2f}  std={np.std(SSE_rand):.2f}")

  ########################
  # Q6 Error vs. K
  ########################
  SSE_vs_k = []
  for k_val in range(1, 151):
    _, _, sse_hist = kMeansClustering(X, k=k_val, max_iters=20)
    SSE_vs_k.append(sse_hist[-1])

  plt.figure(figsize=(16,8))
  plt.plot(range(1,151), SSE_vs_k, marker="o", markersize=3)
  plt.xlabel("k")
  plt.ylabel("SSE")
  plt.title("Q6 – SSE vs. k")
  plt.savefig("q6_sse_vs_k.png", dpi=100)
  plt.close()
  print(f"Q6 elbow region SSEs: k=3:{SSE_vs_k[2]:.2f}  k=5:{SSE_vs_k[4]:.2f}  k=10:{SSE_vs_k[9]:.2f}")


def imageProblem():
  np.random.seed()
  data = np.load("img.npy")
  img_feats = np.load("hog.npy")

  for k in [10, 6]:
    centroids, assignments, SSE = kMeansClustering(img_feats, k, 30, min_size=0)
    print(f"k={k}  final SSE={SSE[-1]:.4f}")

    for c in range(len(centroids)):
      members = np.where(assignments==c)[0].astype(int)
      imgs = data[np.random.choice(members, min(50, len(members)), replace=False),:,:]

      print("Cluster "+str(c)+" ["+str(len(members))+"]")
      _, axs = plt.subplots(5, 10, figsize=(16, 8))
      axs = axs.flatten()
      for img, ax in zip(imgs, axs):
        ax.imshow(img, plt.cm.gray)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
      for i in range(len(imgs), 50):
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
      plt.suptitle(f"k={k}  Cluster {c}  ({len(members)} images)")
      plt.tight_layout()
      plt.savefig(f"q7_k{k}_cluster{c}.png", dpi=60)
      plt.close()


##########################################################
# initializeCentroids
##########################################################
def initalizeCentroids(dataset, k):
  indices = np.random.choice(dataset.shape[0], size=k, replace=False)
  centroids = dataset[indices].copy()
  return centroids

##########################################################
# computeAssignments
##########################################################
def computeAssignments(dataset, centroids):
  # Vectorized squared distances: ||x-c||^2 = ||x||^2 - 2x·c^T + ||c||^2
  x_sq = np.sum(dataset  ** 2, axis=1, keepdims=True)   # (n,1)
  c_sq = np.sum(centroids** 2, axis=1, keepdims=True).T  # (1,k)
  cross = dataset @ centroids.T                          # (n,k)
  dists_sq = x_sq - 2*cross + c_sq                      # (n,k)
  assignments = np.argmin(dists_sq, axis=1)
  return assignments

##########################################################
# updateCentroids
##########################################################
def updateCentroids(dataset, centroids, assignments):
  k = centroids.shape[0]
  new_centroids = np.zeros_like(centroids)
  counts = np.zeros(k, dtype=int)
  for j in range(k):
    mask = assignments == j
    counts[j] = mask.sum()
    if counts[j] > 0:
      new_centroids[j] = dataset[mask].mean(axis=0)
    else:
      new_centroids[j] = centroids[j]   # dead cluster, caller re-inits
  return new_centroids, counts

##########################################################
# calculateSSE
##########################################################
def calculateSSE(dataset, centroids, assignments):
  assigned = centroids[assignments]          # (n,d)
  sse = float(np.sum((dataset - assigned)**2))
  return sse


########################################
# Instructor Code
########################################
def kMeansClustering(dataset, k, max_iters=10, min_size=0, visualize=False):
  centroids = initalizeCentroids(dataset, k)
  SSE = []
  for i in range(max_iters):
    assignments = computeAssignments(dataset, centroids)
    centroids, counts = updateCentroids(dataset, centroids, assignments)
    for c in range(k):
      if counts[c] <= min_size:
        centroids[c] = initalizeCentroids(dataset, 1)
    if visualize:
      plotClustering(centroids, assignments, dataset, "Iteration "+str(i))
    SSE.append(calculateSSE(dataset, centroids, assignments))
  assignments = computeAssignments(dataset, centroids)
  return centroids, assignments, SSE

def plotClustering(centroids, assignments, dataset, title=None):
  plt.figure(figsize=(8,8))
  plt.scatter(dataset[:,0], dataset[:,1], c=assignments, edgecolors="k", alpha=0.5)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=5, edgecolors="k", s=250)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=2, edgecolors="w", s=200)
  if title is not None:
    plt.title(title)

if __name__=="__main__":
  toyProblem()
  imageProblem()
