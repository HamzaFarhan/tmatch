import os

import redis
from dotenv import load_dotenv
from termcolor import colored

load_dotenv()

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)


def connect_to_server(redis_host: str = REDIS_HOST, redis_port: int = REDIS_PORT):
    try:
        r = redis.StrictRedis(
            host=redis_host, port=redis_port, charset="utf-8", decode_responses=True
        )
    except Exception as e:
        raise Exception(f"Connecting to Redis Server failed with error {e}")
    return r


class KeyValueStore:
    def __init__(self, redis_host: str = REDIS_HOST, redis_port: int = REDIS_PORT):
        try:
            self.server = connect_to_server(
                redis_host=redis_host, redis_port=redis_port
            )
            # self.logger = logger
        except Exception:
            raise

    def insert(self, key, value):
        server = self.server
        # ogger.debug(key,value)
        print(colored(f"Inserting key {key} with value {value}", "cyan"))
        server.hmset(key, value)

    def get(self, key):
        server = self.server
        # logger.debug(key)
        try:
            val = server.hgetall(key)
        except Exception as e:
            # logger.error("unable to retrieve value of key {} from Redis: error = {}".format(key,e))
            raise f"Unable to retrieve value of key {key} from Redis: error = {e}"
        return val

    def getall(self):
        return self.server.keys()

    def remove(self, key):
        server = self.server
        # logger.debug('removing key {}'.format(key))
        print(colored(f"Removing key {key}", "cyan"))
        try:
            all_keys = list(server.hgetall(key).keys())
            server.hdel(key, *all_keys)
            # logger.debug('key {} removed'.format(key))
            print(colored(f"Key {key} removed", "cyan"))
        except Exception as e:
            # logger.error("unable to remove key {} from Redis: error = {}".format(key,e))
            raise f"Unable to remove key {key} from Redis: error = {e}"
