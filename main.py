import time

from kmeans_lithops.kmeans.worker import Worker
from kmeans_lithops.kmeans.objects import GlobalDelta, GlobalCentroids
from kmeans_lithops.redis.connection import RedisConn

THRESHOLD = 0.00001
S3_DATAPOINTS_PER_FILE = 695_866  # dataset 100 dimensions
S3_DIMENSIONS = 100


def train(redis_conf,
          worker_id, data_points, dimensions, parallelism,
          clusters, max_iters, debug=False):
    RedisConn.set_up(**redis_conf)
    worker = Worker(worker_id, data_points, dimensions, parallelism,
                    clusters, max_iters, THRESHOLD, debug)
    return worker.run()  # return time info breakdown


def main():
    # #### CONFIGURE ####
    local = False

    redis_conf = {
        'host': 'localhost',
        'port': 6379,
        'password': None
    }

    parallelism = 2
    clusters = 25
    dimensions = S3_DIMENSIONS
    datapoints_per_worker = S3_DATAPOINTS_PER_FILE
    number_of_iterations = 10
    debug = True
    # #### CONFIGURE ####

    RedisConn.set_up(**redis_conf)

    # Initialize global objects
    centroids = GlobalCentroids(clusters, parallelism)
    centroids.random_init(dimensions)
    delta = GlobalDelta(parallelism)
    delta.init()

    worker_stats = []  # in seconds

    def local_run(w_id):
        worker_breakdown = train(redis_conf,
                                 w_id, parallelism * datapoints_per_worker,
                                 dimensions, parallelism, clusters,
                                 number_of_iterations, debug)
        worker_stats.append(worker_breakdown)

    def lambda_run(w_id):
        return train(redis_conf,
                     w_id, parallelism * datapoints_per_worker,
                     dimensions, parallelism, clusters,
                     number_of_iterations, debug)

    if local:
        import threading
        threads = [threading.Thread(target=local_run, args=(i,))
                   for i in range(parallelism)]

        start_time = time.time()
        [t.start() for t in threads]
        [t.join() for t in threads]
        end_time = time.time()
    else:
        import lithops
        ex = lithops.FunctionExecutor()

        start_time = time.time()
        ex.map(lambda_run, range(parallelism))
        results = ex.get_result()
        worker_stats = results
        end_time = time.time()

    # Parse results
    iters_times = []
    for b in worker_stats:
        # Iterations time is second breakdown and last
        iters_times.append(b[-1] - b[2])

    avg_iters_time = sum(iters_times) / len(iters_times)
    print(f"Total k-means time: {end_time - start_time} s")
    print(f"Average iterations time: {avg_iters_time} s")

    import csv
    import sys
    name = str(sys.argv[1])
    with open(name + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(worker_stats)


if __name__ == '__main__':
    main()
