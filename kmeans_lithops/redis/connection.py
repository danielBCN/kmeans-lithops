import redis


class RedisConn(object):
    REDIS = None

    @classmethod
    def set_up(cls, host='localhost', port=6379, password=''):
        cls.REDIS = redis.ConnectionPool(
            host=host, port=port, password=password)
        return cls.get_conn()

    @classmethod
    def get_conn(cls):
        return redis.Redis(connection_pool=cls.REDIS)

