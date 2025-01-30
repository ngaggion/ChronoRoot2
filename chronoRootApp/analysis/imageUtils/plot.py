""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import cv2 
import os
import numpy as np

def getImgName(image, conf):
    return image.replace(conf['ImagePath'],'').replace('/','')

def plot_seg(grafo1, ske):
    g, _, _, clase, _, _ = grafo1
    

    mr = np.zeros_like(ske).astype('uint8')
    lr = np.zeros_like(ske).astype('uint8')
    
    for a in g.get_edges():
        e = g.edge(a[0],a[1])
        pos = np.where(ske==clase[e][0])
        if clase[e][1] == 10:
            mr[pos]=255
        else:
            lr[pos]=255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mr = cv2.dilate(mr, kernel)
    lr = cv2.dilate(lr, kernel)
    
    ske3 = np.zeros(list(ske.shape)+[3], dtype='uint8')
    ske3[:,:,1] = mr
    ske3[:,:,0] = lr 
    
    return ske3

def saveImages(conf, images, i, seg, grafo = None, ske = None):
    folder = conf['folders']['images']
    name = getImgName(images[i], conf)
    bbox = conf['bounding box']

    if conf['saveImages']:
        original = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]

        f1 = os.path.join(folder, "Input")
        path = os.path.join(f1, name)
        cv2.imwrite(path, original)
  
    f2 = os.path.join(folder, "Seg")
    path = os.path.join(f2, name)
    cv2.imwrite(path, seg)

    if grafo is None or ske is None:
        image = np.ones_like(seg).astype('uint8') * 0
    else:
        image = plot_seg(grafo, ske)

    f3 = os.path.join(folder, "SegMulti")
    path = os.path.join(f3, name)
    cv2.imwrite(path, image)
    
    return





        

