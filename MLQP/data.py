from spiral import spiral
import numpy as np
import matplotlib.pyplot as plt

s = spiral(0,1)
x1,y1 = s.out()
z1 = np.zeros(*x1.shape)
d1 = np.vstack((x1,y1,z1))
np.save('s1.npy',d1)

x2,y2 = -x1,-y1
z2 = np.ones(*x2.shape)
d2 = np.vstack((x2,y2,z2))
np.save('s2.npy',d2)

plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'ro')
plt.show()

