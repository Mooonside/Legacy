import numpy as np
import matplotlib.pyplot as plt
from math import cos,sin,pi
#ToDo: Genearte data for 2 spiral

class spiral:
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def out(self):
        #density for how many points in one circle
        t = np.linspace(np.pi/2,2*np.pi*5,200)
        x = (self.a + self.b * t)*np.cos(t)
        y = (self.a + self.b * t)*np.sin(t)
        return x,y
