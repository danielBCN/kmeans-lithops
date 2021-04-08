from ..redis.connection import RedisConn
import numpy as np


class GlobalCentroids(object):
    def __init__(self, clusters, parallelism):
        self.red = RedisConn.get_conn()
        self.num_clusters = clusters
        self.parallelism = parallelism

    @staticmethod
    def centroid_key(centroid):
        return "centroid" + str(centroid)

    def random_init(self, num_dimensions):
        import random
        print(f"Initializing GlobalCentroids with {self.num_clusters},"
              f"{num_dimensions},{self.parallelism}")
        for i in range(self.num_clusters):
            random.seed(1002 + i)
            numbers = [random.gauss(0, 1) for _ in range(num_dimensions)]
            self.red.delete(self.centroid_key(i))
            self.red.rpush(self.centroid_key(i), *numbers)
            # A counter for the updates
            self.red.set(self.centroid_key(i) + "_c", 0)

    def update(self, coordinates, sizes):
        for k in range(self.num_clusters):
            self._update_centroid(k, coordinates[k].tolist(), int(sizes[k]))

    update_centroid_lua_script = """
        local centroidKey = KEYS[1]
        local counterKey = KEYS[2]
        local centroidTemp = KEYS[3]
        local sizeTemp = KEYS[4]
        local n = redis.call("LLEN", centroidKey)
        local count = redis.call("GET", counterKey)
        if count == "0" then
            redis.call("DEL", centroidTemp)
            local a = {}
            for i = 1,2*(n),2 do
                a[i]   = tostring((i-1)/2)
                a[i+1] = tostring(0.0)
            end
            redis.call("HMSET", centroidTemp, unpack(a))
            redis.call("SET", sizeTemp, 0)
        end
        for i = 0,n-1 do
            redis.call("HINCRBYFLOAT", centroidTemp,
                       tostring(i), tostring(ARGV[3+i]))
        end
        local size = redis.call("INCRBY", sizeTemp, ARGV[1])
        count = redis.call("INCR", counterKey)
        if tonumber(count) == tonumber(ARGV[2]) then
            if size ~= "0" then
                redis.call("DEL", centroidKey)
                local temps = redis.call("HGETALL", centroidTemp)
                local values = {}
                for i = 1,n*2,2 do
                    values[(i+1)/2] = temps[i+1]/size
                end
                redis.call("RPUSH", centroidKey, unpack(values))
            end
            redis.call("SET", counterKey, 0)
        end
        return
    """

    def _update_centroid(self, cluster_id, coordinates, size):
        centroid_k = self.centroid_key(cluster_id)
        coordinates = list(map(lambda x: str(x), coordinates))
        self.red.eval(self.update_centroid_lua_script, 4,
                      centroid_k, centroid_k + "_c",
                      centroid_k + "_temp", centroid_k + "_st",
                      size, self.parallelism, *coordinates)

    def get_centroids(self):
        b = [self.red.lrange(self.centroid_key(k), 0, -1)
             for k in range(self.num_clusters)]
        return np.array(
            list(map(lambda point: list(map(lambda v: float(v), point)), b)))


class GlobalDelta(object):
    def __init__(self, parallelism):
        self.red = RedisConn.get_conn()
        self.parallelism = parallelism

    def init(self):
        self.red.set("delta", 1)
        self.red.set("delta_c", 0)
        self.red.set("delta_temp", 0)
        self.red.set("delta_st", 0)

    def get_delta(self):
        return float(self.red.get("delta"))

    update_lua_script = """
        local deltaKey = KEYS[1]
        local counterKey = KEYS[2]
        local deltaTemp = KEYS[3]
        local npointsTemp = KEYS[4]

        local tmpDelta = redis.call("INCRBY", deltaTemp, tostring(ARGV[1]))
        local tmpPoints = redis.call("INCRBY", npointsTemp, ARGV[2])

        local count = redis.call("INCR", counterKey)
        if tonumber(count) == tonumber(ARGV[3]) then
            local newDelta = tmpDelta / tmpPoints
            redis.call("SET", deltaKey, newDelta)
            redis.call("SET", counterKey, 0)
            redis.call("SET", deltaTemp, 0)
            redis.call("SET", npointsTemp, 0)
        end
        return
    """

    def update(self, delta, num_points):
        delta_key = "delta"
        self.red.eval(self.update_lua_script, 4, delta_key, delta_key + "_c",
                      delta_key + "_temp", delta_key + "_st",
                      delta, num_points, self.parallelism)
