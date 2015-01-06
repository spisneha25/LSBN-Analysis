from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import csv, math
from sklearn.cluster import DBSCAN

Distance = []

with open("nodes_list.txt", "rb") as tsv:
    reader = csv.reader(tsv, delimiter = '\t')
    for r in reader:
        Distance.append([float(r[2]), float(r[3])])

map = Basemap(projection='robin', lat_0=-90, lon_0=-180, resolution = 'l', area_thresh = 1000.0)
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='gray')
map.drawmapboundary() 
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))

cluster_size = 100

#DBSCAN
#Perform clustering using DBSCAN
db = DBSCAN(eps = 0.3, min_samples = cluster_size).fit(np.array(Distance))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
lenDistance = len(Distance)

oldRange = (1 - cluster_size/float(lenDistance))
newRange = (20 - 5)            
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
    class_member_mask = (labels == k)
    
    xy = np.array(Distance)[class_member_mask & core_samples_mask]

    if xy.any():
        mark = (len(xy)/float(len(Distance)) - cluster_size/float(lenDistance)) * newRange * 15 / oldRange + 5
        x, y = map(np.mean(xy[:, 1]), np.mean(xy[:, 0]))
        map.plot(x, y, "ro", markersize = abs(mark))
plt.show()


       

