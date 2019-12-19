import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as sl
import math 
from heapq import heapify, heappush, heappop
import copy
from mpl_toolkits.mplot3d import Axes3D

file = open('D:\Documents\Data Mining\Project\\seeds_dataset.txt')
data = file.readlines()
for i in range (0,len(data)):
    data[i] = data[i].strip().split()
    for j in range(0,len(data[i])):
        data[i][j] = float(data[i][j])
    
dataarray = np.array(data).transpose()
for i in range(0,len(dataarray) - 1):
    maxd = np.amax(dataarray[i])
    mind = np.amin(dataarray[i])
    for j in range(0,len(dataarray[i])):
        dataarray[i][j] = -1 + (2 * (dataarray[i][j] - mind))/(maxd - mind)

dataarray = dataarray.transpose()
data = dataarray.tolist()
attrisbutes = ["Area", "Perimeter", "Compactness", "Kernel Length", "Kernel Width", "Assymetry coefficient", "Kernel groove length"]
colorpos = ['red','green', 'blue']
colors = [colorpos[int(data[i][-1]) - 1] for i in range(len(data))]
reachdist = [-1 for _ in range(len(data))]
proceslist = [False for _ in range(len(data))]
def Optics(Data, eps, minpts, i = -1, j = -1, seeds = []):
    ordlist = []
    for p in range(len(Data)):
        if proceslist[p] == False:
            N = getNeighbors(Data,Data[p],eps,i,j)
            proceslist[p] = True
            ordlist.append(p)
            if core_distance(Data, Data[p], minpts, N, i, j) > 0:
                seeds = priority_dict(dict(seeds))
                update(Data, N,Data[p],seeds,eps,minpts,i,j)
                iter = seeds.sorted_iter()
                for q in iter:
                    Nprime = getNeighbors(Data,Data[int(q)],eps,i,j)
                    proceslist[int(q)] = True
                    ordlist.append(int(q))
                    if core_distance(Data,Data[int(q)],minpts,N, i,j) > 0:
                        update(Data,Nprime,Data[int(q)],seeds,eps,minpts,i,j)
    array = []
    for p in ordlist:
        array.append((reachdist[p],p))
    fig, (ax1, ax2) = plt.subplots(2,1)
    if i != -1 and j != -1:
        objects = [i for i in range(len(data))]
        ilist = []
        jlist = []
        for p in objects:
            ilist.append(Data[p][i])
            jlist.append(Data[p][j])
        ax1.scatter(ilist,jlist, color = colors)
        ax1.scatter(ilist[0], jlist[0], color = "green", label = "Rosa wheat")
        ax1.scatter(ilist[0], jlist[0], color = "blue" , label = "Canadian wheat")
        ax1.scatter(ilist[0], jlist[0], color = "red", label = "Kama wheat")
        ax1.legend()
        ax1.set_title("scatter plot of "+ attrisbutes[i] + " vs " + attrisbutes[j])
        ax1.set_xlabel(attrisbutes[i])
        ax1.set_ylabel(attrisbutes[j])
    xreach = [i for i in range(1,len(Data))]
    yreach = [j[0] for j in array[1:]]
    ax2.scatter(xreach, yreach, color = [colors[q[1]] for q in array[1:]])
    ax2.scatter(xreach[0], yreach[0], color = "green", label = "Rosa wheat")
    ax2.scatter(xreach[0], yreach[0], color = "blue" , label = "Canadian wheat")
    ax2.scatter(xreach[0], yreach[0], color = "red", label = "Kama wheat")
    ax2.set_title("Reachability plot eps = " + str(eps) + ", minPts = " + str(minpts))
    ax2.set_ylabel("Reachability distance")
    ax2.legend()
    plt.show()
    return ordlist
        

def dist(Data,p,q,i,j):
    if i == -1 and j == -1:
        dist = 0
        for h in range(len(p) - 1):
            dist = dist + (p[h] - q[h])**(2)
        dist = math.sqrt(dist)
        return dist
    else:
        dist = 0
        dist = math.sqrt((p[i] - q[i])**(2) + (p[j] - q[j])**(2))
        return dist        
            
            
def getNeighbors(Data,p,eps,i,j):
    afstand = []
    for n in range(len(Data)):
        afstand.append((0,0))
        distance = dist(Data,p,Data[n],i,j)
        afstand[-1] = (distance, n)
    afstand.sort()
    neighbors = [Data[q[1]] for q in afstand if q[0] < eps and q[0] != 0]
    return neighbors


def core_distance(Data, p, minpts, N, i, j):
    if len(N) < minpts:
        return -1
    else: 
        return dist(Data,p, N[minpts - 1], i, j)
        
def update(Data, N, p, seeds, eps, minpts , i ,j):
    coredist = core_distance(Data, p, minpts, N, i, j)
    for o in range(len(Data)):
        if proceslist[o] == False:
            new_reach_dist = max(coredist, dist(Data,p,Data[o], i, j))
            if reachdist[o] < 0:
                reachdist[o] = new_reach_dist
                seeds[str(o)] = new_reach_dist
            else:
                if new_reach_dist < reachdist[o]:
                    reachdist[o] = new_reach_dist
                    seeds[str(o)] = new_reach_dist
                    
class priority_dict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """
    
    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """
        
        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).
        
        super(priority_dict, self).__setitem__(key, val)
        
        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.
        
        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """
        
        while self:
            yield self.pop_smallest()

                

        

        
        
        