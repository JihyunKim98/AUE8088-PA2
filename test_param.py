import numpy as np
import glob
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans

def load_dataset(path):
    dataset = []
    for xml_file in glob.glob(path):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)

def kmeans_anchors(path, num_anchors=9):
    dataset = load_dataset(path)
    kmeans = KMeans(n_clusters=num_anchors)
    kmeans.fit(dataset)
    anchors = kmeans.cluster_centers_
    return anchors

# Example usage
path = '/home/jihyun/dataset/annotations/*.xml'
anchors = kmeans_anchors(path)
print("Best anchors:", anchors)
