import boto3
import numpy as np


class S3Reader(object):
    S3_BUCKET = "gparis-kmeans-dataset"

    def get_points(self, worker_id, partition_points, num_dimensions):
        file_name = "dataset-100GB-100d/part-" + "{:05}".format(worker_id)
        print(f"s3BUCKET::::::: {self.S3_BUCKET}")

        s3 = boto3.resource('s3')
        # s3 = boto3.resource('s3', aws_access_key_id='xxx',
        #                     aws_secret_access_key='xxx')
        obj = s3.Object(self.S3_BUCKET, file_name).get()['Body']

        points = np.zeros([partition_points, num_dimensions])

        lines = 0
        for line in obj.iter_lines():
            dims = (line.decode("utf-8")).split(',')
            # points.append(list(map(lambda x: float(x), dims)))
            points[lines] = np.array(list(map(lambda x: float(x), dims)))
            lines += 1

        obj.close()
        print(f"Dataset loaded from file {file_name}")
        print(f"First point: {points[0][0]}  "
              f"Last point: {points[partition_points - 1][num_dimensions - 1]}")
        print(f"Points loaded: {points.shape[0]}")

        for p, point in enumerate(points):
            if len(point) != num_dimensions:
                print(f"Worker {worker_id} Reading ERROR: point {p} "
                      f"only has {len(point)} dimensions!")
        return points
