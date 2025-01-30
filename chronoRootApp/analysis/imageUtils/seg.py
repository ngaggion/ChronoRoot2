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

from skimage.morphology import skeletonize
import cv2
import numpy as np
import cv2

def getCleanSeg(segFile, bbox, seed, originalSeed):
    # Loads the segmentation file
    seg = cv2.imread(segFile, 0)[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    
    # Removes all the segmentations over the seed point
    seg[0:originalSeed[1],:] = 0

    seg = (seg == 1) + (seg == 2)
    seg = np.array(seg, dtype = 'uint8') * 255

    # Opening and closing morphological operations
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    seg = cv2.dilate(seg, kernel)
    seg = cv2.erode(seg, kernel)
    seg = cv2.erode(seg, kernel)
    seg = cv2.dilate(seg, kernel)
    
    # Finding the different connected components, in order of size
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    j = 0

    if len(contour_sizes) != 0:
        ### Sorts list of contours by size from bigger to smaller
        contour_sizes.sort(key=lambda x: x[0], reverse=True)
        
        for contour in contour_sizes:
            if contour[0] < 30:
                break
            else:
                dist = cv2.pointPolygonTest(contour[1],(int(seed[0]), int(seed[1])), True)
                dist = np.abs(dist)
                is_in = cv2.pointPolygonTest(contour[1],(int(seed[0]), int(seed[1])), False) > 0
                
                if (dist < 30 or is_in):
                    mask = np.zeros(seg.shape, np.uint8)
                    cv2.drawContours(mask,[contour[1]], -1, 255, -1) 
                    seg2 = cv2.bitwise_and(mask, seg.copy())
                
                    return seg2, True
            
    return seg, False


def getCleanSke(seg):
    ske = np.array(skeletonize(seg // 255), dtype = 'uint8')
    
    contours, _ = cv2.findContours(ske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ske = prune(ske, 5)
    ske = trim(ske)
    ske = prune(ske, 3)
    ske = trim(ske)

    contours, _ = cv2.findContours(ske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bnodes, enodes = skeleton_nodes(ske)
    
    flag = False
    if len(enodes) >= 2:
        flag = True
        
    return ske, bnodes, enodes, flag

def trim(ske): ## Removes unwanted pixels from the skeleton
    
    T=[]
    T0=np.array([[-1, 1, -1], 
                 [1, 1, 1], 
                 [0, 0, 0]]) # T0 contains X0
    T2=np.array([[-1, 1, 0], 
                 [1, 1, 0], 
                 [-1, 1, 0]])
    T4=np.array([[0, 0, 0], 
                 [1, 1, 1], 
                 [-1, 1, -1]])
    T6=np.array([[0, 1, -1], 
                 [0, 1, 1], 
                 [0, 1, -1]])
    S1=np.array([[1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, -1]])
    S2=np.array([[-1, 1, -1], 
                 [1, 1, -1], 
                 [1, -1, -1]])
    S3=np.array([[-1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, 1]])
    S4=np.array([[-1, -1, -1], 
                 [-1, 1, 1], 
                 [1, 1, -1]])
    S5=np.array([[-1, 1, 1], 
                 [1, 1, -1], 
                 [-1, -1, -1]])
    S6=np.array([[1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, -1]])
    S7=np.array([[-1, -1, 1], 
                 [-1, 1, 1], 
                 [-1, 1, -1]])
    S8=np.array([[-1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, 1]])
    C1=np.array([[-1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, -1]])
    C2=np.array([[-1, -1, -1], 
                 [-1, 1, 1], 
                 [-1, 1, -1]])
    C3=np.array([[-1, 1, -1], 
                 [1, 1, -1], 
                 [-1, -1, -1]])
    C4=np.array([[-1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, -1]])
    
    T.append(T0)
    T.append(T2)
    T.append(T4)
    T.append(T6)
    T.append(S1)
    T.append(S2)
    T.append(S3)
    T.append(S4)
    T.append(S5)
    T.append(S6)
    T.append(S7)
    T.append(S8)    
    T.append(C1)
    T.append(C2)
    T.append(C3)
    T.append(C4)
    
    bp = np.zeros_like(ske)
    for t in T:
        bp = cv2.morphologyEx(ske, cv2.MORPH_HITMISS, t)
        ske = cv2.subtract(ske, bp)
    
    # ske = cv2.subtract(ske, bp)
    
    return ske


def prune(skel, num_it): ## Removes branches with length lower than num_it
    orig = skel
    
    endpoint1 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [0, 1, 0]])
    
    endpoint2 = np.array([[0, 1, 0],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    endpoint4 = np.array([[0, -1, -1],
                          [1, 1, -1],
                          [0, -1, -1]])
    
    endpoint5 = np.array([[-1, -1, 0],
                          [-1, 1, 1],
                          [-1, -1, 0]])
    
    endpoint3 = np.array([[-1, -1, 1],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    endpoint6 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [1, -1, -1]])
    
    endpoint7 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [-1, 1, -1]])
    
    endpoint8 = np.array([[1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    
    for i in range(0, num_it):
        ep1 = skel - cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint1)
        ep2 = ep1 - cv2.morphologyEx(ep1, cv2.MORPH_HITMISS, endpoint2)
        ep3 = ep2 - cv2.morphologyEx(ep2, cv2.MORPH_HITMISS, endpoint3)
        ep4 = ep3 - cv2.morphologyEx(ep3, cv2.MORPH_HITMISS, endpoint4)
        ep5 = ep4 - cv2.morphologyEx(ep4, cv2.MORPH_HITMISS, endpoint5)
        ep6 = ep5 - cv2.morphologyEx(ep5, cv2.MORPH_HITMISS, endpoint6)
        ep7 = ep6 - cv2.morphologyEx(ep6, cv2.MORPH_HITMISS, endpoint7)
        ep8 = ep7 - cv2.morphologyEx(ep7, cv2.MORPH_HITMISS, endpoint8)
        skel = ep8
        
    end = endPoints(skel)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    
    for i in range(0, num_it):
        end = cv2.dilate(end, kernel)
        end = cv2.bitwise_and(end, orig)
        
    return cv2.bitwise_or(end, skel)


def endPoints(skel):
    endpoint1=np.array([[1, -1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint2=np.array([[-1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint3=np.array([[-1, -1, 1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint4=np.array([[-1, -1, -1],
                        [1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint5=np.array([[-1, -1, -1],
                        [-1, 1, 1],
                        [-1, -1, -1]])
    
    endpoint6=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [1, -1, -1]])
    
    endpoint7=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [-1, 1, -1]])
    
    endpoint8=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1]])
    
    ep1 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint1)
    ep2 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint2)
    ep3 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint3)
    ep4 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint4)
    ep5 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint5)
    ep6 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint6)
    ep7 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint7)
    ep8 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint8)
    
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep


def skeleton_nodes(ske):
    branch = branchedPoints(ske)
    end = endPoints(ske)
    
    bp = np.where(branch == 1)
    bnodes = []
    for i in range(len(bp[0])):
        bnodes.append([bp[1][i],bp[0][i]])
    
    ep = np.where(end == 1)
    enodes = []
    for i in range(len(ep[0])):
        enodes.append([ep[1][i],ep[0][i]])
    
    return np.array(bnodes), np.array(enodes)


def branchedPoints(skel):
    X=[]
    #cross X
    X0 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    X1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    
    #T like
    T=[]
    T0=np.array([[2, 1, 2], 
                 [1, 1, 1], 
                 [2, 2, 2]]) # T0 contains X0
    T1=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]]) # T1 contains X1
    T2=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    T3=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    T4=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    T5=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    T6=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    T7=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    
    #Y like
    Y=[]
    Y0=np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]])
    Y1=np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]])
    Y2=np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]])
    Y3=np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]])
    Y4=np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
    Y5 = np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)
    
    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, x)
    for y in Y:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, y)
    for t in T:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, t)
        
    return bp