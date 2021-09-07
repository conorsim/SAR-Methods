from Image import Image
import numpy as np
import rasterio
import gc

class Cluster(Image):

    def __init__(self, c_bounds):
        self.c_bounds = c_bounds

    """ Driver to process ISODATA """

    # based on http://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Vassilaros_ISODATA.pdf

    # img - band of an Image object
    def isodata(img, random_init=False, start_num=20, max_iter=10, std_threshold=0.1, merge_threshold=0.1, \
                size_threshold=50):

        # initialize starting cluster centers
        if random_init: centers = [np.random.uniform(size=3) for _ in range(start_num)]
        else:
            rows = [np.random.choice(range(img.shape[0]), replace=False) for _ in range(start_num)]
            cols = [np.random.choice(range(img.shape[1]), replace=False) for _ in range(start_num)]
            centers = [img[row, col] for row,col in list(zip(rows, cols))]
        center_mats = []
        for center in centers:
            center_mat = np.ones((img.shape[0], img.shape[1], 3))
            center_mat[:,:,:] = center
            center_mats.append(center_mat)
        dist_mats = np.zeros((img.shape[0], img.shape[1], len(centers)))
        for k in range(len(centers)):
            center_mat = center_mats[k]
            dist_mat = np.linalg.norm(img-center_mat, axis=2) # Euclidean distance between each pixel and a certain cluster
            dist_mats[:,:,k] = dist_mat
        assignments = np.argmin(dist_mats, axis=2)
        cluster_ids = np.unique(assignments)
        # free memory
        del center_mats
        del dist_mats
        gc.collect()

        # iterate
        count = 0
        cont = True
        while cont:
            # perform merge test
            n = len(cluster_ids)
            neg_count = -1
            for i in range(n):
                for j in range(i+1,n):
                    id1 = cluster_ids[i]
                    id2 = cluster_ids[j]
                    if id1 > 0 and id2 > 0: # if the clusters have not already been merged
                        c1 = centers[id1]
                        c2 = centers[id2]
                        inter_dist = np.linalg.norm(c1-c2)
                        if inter_dist < merge_threshold: # merge clusters that have centers that are close together
                            print(f'Merging clusters {id1} and {id2} into cluster {neg_count}')
                            cluster_ids[i] = neg_count
                            cluster_ids[j] = neg_count
                            assignments = np.where(assignments == id1, neg_count, assignments)
                            assignments = np.where(assignments == id2, neg_count, assignments)
                            neg_count -= 1

            # perform standard deviation test and re-define centers
            centers = []
            cluster_ids = np.unique(cluster_ids) # collapse duplicates from merged clusters
            for cluster_id in cluster_ids:
                rows, cols = np.where(assignments == cluster_id)
                pixels = img[rows, cols, :]
                std = np.std(pixels, axis=0, ddof=1)
                test_std = np.where(std > std_threshold, 1, 0)
                if np.sum(test_std) > 0:
                    # split the cluster
                    print(f'Splitting cluster number {cluster_id} with std {np.max(std)}')
                    faulty_dim = np.argmax(std) # choose the dimension with the highest std
                    dim_vec = np.array(pixels[:, faulty_dim])
                    cut_off = np.quantile(dim_vec, 0.5)
                    l = np.where(dim_vec <= cut_off)[0]
                    sub_pix1 = pixels[l,:]
                    u = np.where(dim_vec > cut_off)[0]
                    sub_pix2 = pixels[u,:]
                    centers.append(np.mean(sub_pix1, axis=0))
                    centers.append(np.mean(sub_pix2, axis=0))
                else:
                    centers.append(np.mean(pixels, axis=0))

            # re-assign the clusters
            center_mats = []
            for center in centers:
                center_mat = np.ones((img.shape[0], img.shape[1], 3))
                center_mat[:,:,:] = center
                center_mats.append(center_mat)
            dist_mats = np.zeros((img.shape[0], img.shape[1], len(centers)))
            for k in range(len(centers)):
                center_mat = center_mats[k]
                dist_mat = np.linalg.norm(img-center_mat, axis=2) # Euclidean distance between each pixel and a certain cluster
                dist_mats[:,:,k] = dist_mat
            assignments = np.argmin(dist_mats, axis=2)
            cluster_ids = np.unique(assignments)
            # free memory
            del center_mats
            del dist_mats
            gc.collect()

            # stop if maximum iterations is reached
            count += 1
            if count >= max_iter: cont = False

        return assignments
