import time

import numpy as np
from scipy.spatial.distance import cdist

from .objects import GlobalCentroids, GlobalDelta
from .s3reader import S3Reader
from ..redis.barrier import RedisBarrier


class Worker(object):

    def __init__(self, worker_id, data_points, dimensions, parallelism,
                 clusters, max_iters, delta_threshold,
                 debug=False, s3_data=False):
        self.worker_id = worker_id
        self.num_dimensions = dimensions
        self.num_clusters = clusters
        self.max_iterations = max_iters
        self.delta_threshold = delta_threshold
        self.partition_points = int(data_points / parallelism)
        self.parallelism = parallelism
        self.start_partition = self.partition_points * self.worker_id
        self.end_partition = self.partition_points * (worker_id + 1)
        self.debug = debug
        self.s3_data = s3_data

        self.correct_centroids = None
        self.local_partition = None
        self.local_partition_size = 0
        self.local_centroids = None
        self.local_sizes = None
        self.local_membership = None

        self.barrier = RedisBarrier("barrier", self.parallelism)
        self.global_delta = GlobalDelta(self.parallelism)
        self.global_centroids = GlobalCentroids(self.num_clusters,
                                                self.parallelism)

    def run(self):
        print(f"Thread {self.worker_id}/{self.parallelism} with "
              f"k={self.num_clusters} maxIts={self.max_iterations}")

        self.global_centroids = GlobalCentroids(self.num_clusters,
                                                self.parallelism)
        breakdown = [time.time()]

        self.load_dataset()
        if self.debug:
            print(self.local_partition)

        self.local_membership = np.zeros([self.local_partition_size])

        # barrier before starting iterations, to avoid different execution times
        self.barrier.wait()

        breakdown.append(time.time())
        iter_count = 0
        global_delta_val = 1
        while (iter_count < self.max_iterations) and \
                (global_delta_val > self.delta_threshold):
            if self.debug:
                print(f"Iteration {iter_count} of worker {self.worker_id}")

            # Get local copy of global objects
            self.correct_centroids = self.global_centroids.get_centroids()
            breakdown.append(time.time())

            # Reset data structures that will be used in this iteration
            self.local_sizes = np.zeros([self.num_clusters])
            self.local_centroids = np.zeros(
                [self.num_clusters, self.num_dimensions])
            if self.debug:
                print("Structures reset")

            # Compute phase, returns number of local membership modifications
            delta = self.compute_clusters()
            breakdown.append(time.time())
            if self.debug:
                print(f"Compute finished in {breakdown[-1] - breakdown[-2]} s")

            # Update global objects
            self.global_delta.update(delta, self.local_partition_size)
            self.global_centroids.update(
                self.local_centroids, self.local_sizes)
            breakdown.append(time.time())

            p = self.barrier.wait()
            if self.debug:
                print(f"Await: {p}")
            breakdown.append(time.time())

            global_delta_val = self.global_delta.get_delta()
            if self.debug:
                print(f"DEBUG: Finished iteration {iter_count} of worker "
                      f"{self.worker_id} [GlobalDeltaVal={global_delta_val}]")
            iter_count += 1

        breakdown.append(time.time())
        iteration_time = breakdown[-1] - breakdown[0]
        print(f"{iter_count} iterations in {iteration_time} s")
        return breakdown
        # Breakdown: [run start, initial after first barrier,
        #             after get global centroids, after compute,
        #             after global update, after barrier,
        #             ... (next iters including get global delta),
        #             final]

    def load_dataset(self):
        if self.s3_data:
            self.local_partition = S3Reader().get_points(self.worker_id,
                                                         self.partition_points,
                                                         self.num_dimensions)
        else:
            self.local_partition = np.random.randn(self.partition_points,
                                                   self.num_dimensions)
        self.local_partition_size = self.local_partition.shape[0]

    def compute_clusters(self):
        points = self.local_partition
        centroids = self.correct_centroids

        dists = cdist(points, centroids, 'sqeuclidean')
        # for each point, id of closest cluster
        min_dist_cluster_id = dists.argmin(1)  # aka memberships

        # count of points to each cluster
        self.local_sizes = np.bincount(min_dist_cluster_id,
                                       minlength=self.num_clusters)

        # sum of points to each cluster
        cluster_ids = np.unique(min_dist_cluster_id)
        for cluster_id in cluster_ids:
            points[min_dist_cluster_id == cluster_id] \
                .sum(axis=0, out=self.local_centroids[cluster_id])

        # check changes in membership
        new_memberships = min_dist_cluster_id
        delta = np.count_nonzero(
            np.add(new_memberships, - self.local_membership))
        self.local_membership = new_memberships
        return delta
