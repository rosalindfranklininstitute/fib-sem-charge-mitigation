'''
Copyright <2024> Rosalind Franklin Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_minimum

plt.close('all')


"Flyback distortions are estimated from acquisitions using interleaved scan with shortest dwell time"
#These acquisitions suffer more from distortions because of the short dwell time
#The extension of the distortions is statistically estimated from multiple sequential acquisitions
#Distortions starts to settled down after 50 pixels approx.
#90 pixels is selected here as the width of a small region from where flyback distortions are estimated
width = 90
#Draw a rectangle to highlight this region, coordinates below:
xs = [0, width, width, 0, 0]
y1 = 700
y2 = 1000
ys = [y1, y1, y2, y2, y1]

"To store the extension of distortions estimated for each of the frames"
maximos = []

"Estimate the distortions from each of the frames acquired at 100 ns, 100 frames"
total_frames = 100
for frame in range(0, total_frames):
        
    "Interleaved test data"
    imagen = np.load(r'D:\QDscan_engine\SEM\nt26276-529\RP1\100ns\Interleaved\IL_FI_' + str(frame) + '.npy')
    
    "Crop a small region on the left, shift pixel intensities to positive values and normalize for Gaussian filter"
    imagencropped = imagen[y1: y2, 0: width]
    imagencropped = imagencropped.astype(np.int32)
    imagencropped = imagencropped - (-32768)
    imagencropped = imagencropped.astype(np.uint16)
    imagencropped = (imagencropped - np.min(imagencropped))/(np.max(imagencropped)-np.min(imagencropped))
    
    "Apply Gaussian filter to denoise"
    imagenfiltered = ndimage.gaussian_filter(imagencropped, 7)#increase sigma if threshold_minimum fails to find a suitable threshold
    
    
    "Apply sobel filter and threshold for edge detection"
    temp = np.zeros((imagenfiltered.shape)) 
    sx = ndimage.sobel(imagenfiltered, axis = 0)#Sobel filter, x direction
    sy = ndimage.sobel(imagenfiltered, axis = 1)#Sobel filter, y direction
     
    temp = np.hypot(sx, sy)
    
    thres = threshold_minimum(temp)
    temp = temp > thres
    
    if frame==0 or frame==15 or frame==23 or frame==56 or frame==79 or frame==88:
        plt.figure()
        plt.subplot(131)
        plt.imshow(imagen, cmap= 'gray')
        plt.plot(xs, ys, color="yellow", linewidth = 1.8, linestyle = "dashed")#draw a rectangle
        plt.axis('off')
        plt.title('Frame = ' + str(frame))

        plt.subplot(132)
        plt.imshow(imagenfiltered)
        plt.title('Gaussian filtered')
    
        plt.subplot(133)
        plt.imshow(temp)
        plt.title('Sobel filtered and thresholded')
        
    #estimate the height of the bands
    suma = np.sum(temp, axis = 0)
    #consider only cropped images where vertical bands have a continuity of more than 1/4th of the height of these regions
    if np.asarray(np.where(suma > (y2-y1)//4)).any() > 0:
        maximos = np.append(maximos, np.max(np.where(suma > (y2-y1)//4)))#the maximum of suma estimates the right edge of the band


plt.figure()
plt.hist(maximos, color = 'coral', bins = 15)
plt.ylabel('Frequency', fontsize ='14')
plt.xlabel('Pixel position', fontsize = '14')
print('Percentage of images employed to calculate the histogram = ' + str(100*len(maximos)//total_frames) + '%')



