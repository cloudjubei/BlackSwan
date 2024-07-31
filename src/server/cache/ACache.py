from abc import ABC, abstractmethod
from typing import List

class ACache(ABC):
    def __init__(self):
        super().__init__()

        self.cache = {}
        self.cacheSize = 10

    def setup(self, keys: List[str], internalKeys: List[str], cacheSize: int):
        self.cacheSize = cacheSize
        for key in keys:
            self.cache[key] = {}
            for internalKey in internalKeys:
                self.cache[key][internalKey] = []

    def setupValues(self, key: str, internalKey: str, values: List):
        self.cache[key][internalKey] = values.reverse()

    def store(self, key: str, internalKey: str, value):
        self.cache[key][internalKey] = [value] + self.cache[key][internalKey]
        if len(self.cache[key][internalKey]) > self.cacheSize:
            self.cache[key][internalKey].pop()

    def getAllKeys(self) -> List[str]:
        return list(self.cache.keys())
    
    def getAllInternalKeys(self, key: str) -> List[str]:
        return list(self.cache[key].keys())

    def getAll(self, key: str, internalKey: str) -> List:
        return self.cache[key][internalKey] or []

    def getLatest(self, key: str, internalKey: str):
        if len(self.cache[key][internalKey])  > 0:
            return self.cache[key][internalKey][0]

class PriceCache(ACache):

    def __init__(self):
        super(ACache, self).__init__()

    def storePrice(self, value): #PriceModel
        latest = self.getLatest(value.tokenPair, value.interval)
        if latest.timestamp > value.timestamp:
            self.store(value.tokenPair, value.interval, value)
        else:
            self.cache[value.tokenPair][value.interval][0] = value
