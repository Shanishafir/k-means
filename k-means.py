import matplotlib.pyplot as plt
import numpy
import numpy as np
import sys


def k_means(k, n, f, u, pixels, z):
    # new centroids array.
    new_z = []
    # initial clusters array.
    clusters = []
    for t in range(0, k):
        clusters.append([])
    # Checking each pixel.
    for i in range(0, n):
        minimal_distance = 3
        # The index of the centroid with the minimum distance to this pixel.
        minimal_index = 0
        # Searching the centroid with the minimum distance to the pixel.
        for j in range(0, k):
            current_distance = np.linalg.norm(pixels[i] - z[j])
            if current_distance < minimal_distance:
                minimal_distance = current_distance
                minimal_index = j
        clusters[minimal_index].append(pixels[i])
    # Compute the average of each cluster and update the centroids.
    for p in range(0, k):
        sum_of_cluster = sum(clusters[p])
        new_z.append(sum_of_cluster / (max(len(clusters[p]), 1)))
        new_z[p] = np.around(new_z[p], 4)
    # Checking if convergence occurred.
    flg = False
    for i in range(0, len(new_z)):
        if type(new_z[i]) != numpy.ndarray: new_z[i] = np.array([0., 0., 0.]); continue
        if flg:
            continue
        for j in range(0, len(new_z[i])):
            if new_z[i][j] != z[i][j]:
                flg = True

    z = new_z
    f.write(f"[iter {u}]:{','.join([str(i) for i in new_z])}" + "\n")
    new_z = []

    if not flg:
        exit()
    return z

def main():
    image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
    z = np.loadtxt(centroids_fname)  # load centroids
    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255.
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    pixels = pixels.reshape(-1, 3)
    k = len(z)
    n = len(pixels)
    f = open(out_fname, "w+")
    for u in range(20):
        z = k_means(k, n, f, u, pixels, z)
    f.close()

if __name__ == '__main__':
    main()