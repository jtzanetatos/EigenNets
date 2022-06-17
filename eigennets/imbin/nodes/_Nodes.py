# -*- coding: utf-8 -*-
"""

"""

from numpy import zeros, uint8, uint32
from numpy import float32 as npfloat32
from numba.experimental import jitclass
from numba import int32, float32

# ========================================================================== #
# TODO:  implement proper spec, implement proper init
# spec = [
#         ('value', int32),
#         ('array', float32)
# ]

# @jitclass(spec)
class nodes(object):
    
    def __init__(self, cluster_size):
        self.covariance = zeros((3, 3), dtype=npfloat32)
        self.intensity = zeros(3, dtype=uint32)
        self.cluster_size = cluster_size
        self.point = zeros(cluster_size, dtype=uint32)
        self.eigvec = zeros(3, dtype=npfloat32)
        self.eigvals = float32(0)
        self.entropy = float32(0.0)
        self.parent = uint8(0)
        self.itr = uint8(0)
        self.children = None
        self.nodeself = uint8(0)
        self.childIter = None
        
    @property
    def covariance(self):
        return self.__covariance
    
    @covariance.setter
    def covariance(self, covariance):
        self.__covariance = covariance
    
    @property
    def intensity(self):
        return self.__intensity
    
    @intensity.setter
    def intensity(self, intensity):
        self.__intensity = intensity
    
    @property
    def cluster_size(self):
        return self.__cluster_size
    
    @cluster_size.setter
    def cluster_size(self, cluster_size):
        self.__cluster_size = cluster_size
    
    @property
    def point(self):
        return self.__point
    
    @point.setter
    def point(self, point):
        self.__point = point
    
    @property
    def eigvec(self):
        return self.__eigvec
    
    @eigvec.setter
    def eigvec(self, eigvec):
        self.__eigvec = eigvec
    
    @property
    def eigvals(self):
        return self.__eigvals
    
    @eigvals.setter
    def eigvals(self, eigvals):
        self.__eigvals = eigvals
    
    @property
    def entropy(self):
        return self.__entropy
    
    @entropy.setter
    def entropy(self, entropy):
        self.__entropy = entropy
    
    @property
    def parent(self):
        return self.__parent
    
    @parent.setter
    def parent(self, parent):
        self.__parent = parent
    
    @property
    def itr(self):
        return self.__itr
    
    @itr.setter
    def itr(self, itr):
        self.__itr = itr
    
    @property
    def children(self):
        return self.__children
    
    @children.setter
    def children(self, children):
        self.__children = children
    
    @property
    def nodeself(self):
        return self.__nodeself
    
    @nodeself.setter
    def nodeself(self, nodeself):
        self.__nodeself = nodeself
    
    @property
    def childIter(self):
        return self.__childIter
    
    @childIter.setter
    def childIter(self, childIter):
        self.__childIter = childIter
    
    def __str__(self):
        return "Node covariance: " + str(self.covariance)+"\n"+ \
            "Node eigenvector: "+str(self.eigvec)+"\n"+\
            "Node intensity: "+str(self.intensity)+\
            "\nCluster size: %d\nEigenvalue: %d"%(self.cluster_size,
                                                  self.eigvals)+\
            "\nNode Entropy: %f\nNode parent: %d\nNode self: %d\n"%(self.entropy,
                                                                     self.parent,
                                                                     self.nodeself)+\
            "Iteration: %d\nNodes' children: "%(self.itr)+str(self.children)+"\n"+\
            "Nodes' children iteration: "+str(self.childIter)
    
    def __repr__(self):
        return self.__str__()