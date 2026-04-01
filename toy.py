

import numpy as np
rng = np.random.default_rng(seed=42)

from collections import deque

buffer=deque()
buffer.append((1,2,3))
buffer.append((4,5,6))
buffer.append((3,2,1))

e=rng.choice(buffer,size=2,replace=True)




