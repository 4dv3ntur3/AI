#shuffle

import numpy as np
import pandas as pd

sf = np.load('./data/sdss_uv.npy', allow_pickle=True)
np.random.shuffle(sf)
np.save('./data/sdss_uv_s.npy', arr=sf)