from data_loader import TSPDataLoader
from config import get_config
import numpy as np 

rng = np.random.RandomState(123)
config, unparsed = get_config()
tsp = TSPDataLoader(config, rng)