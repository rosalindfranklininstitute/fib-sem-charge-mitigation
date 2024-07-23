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

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


"Define frame size and number of pixels to skip"
xsize = 32
ysize = xsize
skip = 2 #raster scan would be -> skip = 0


fraction = (1/(skip+1))*(1/(skip+1)) #fraction of scan positions per subframe
print('Fraction of scan position for each subframe = ', fraction)
subframes = int(1/fraction) #number of subframes to scan all the positions 
print('Number of required subframes = ', subframes)


"1D arrays to store scan positions (x and y) following the interleaved (or raster if skip = 0) pattern"
scanposx = []
scanposy = []



"Create pattern"
i = 1
for first_row in range(0, int(np.sqrt(subframes))):
    for first_column in range(0, int(np.sqrt(subframes))):
        A = np.zeros((ysize, xsize), dtype = int)
        B = np.zeros((ysize, xsize), dtype = int)
        
        A[0:ysize, first_column::(skip+1)] = 1 #skip columns starting from first_column
        B[first_row::(skip+1), 0:xsize] = 1 #skip rows starting from first_row
        
        "Show subframes"
        plt.figure()
        plt.imshow(A*B)
        plt.title('Subframe ' + str(i))
        i = i + 1
        
        scanposx = np.concatenate((scanposx, np.nonzero(A*B)[1]))
        scanposy = np.concatenate((scanposy, np.nonzero(A*B)[0]))


"x and y final scan positions"
scanposx = scanposx.astype(int)
scanposy = scanposy.astype(int)



"Save positions in a text file (.xy)"
with open("interleaved32x32.xy", "w") as f:
    for i in range(len(scanposx)):
        f.write(str(scanposx[i]) + " , " + str(scanposy[i]) + "\n")
f.close()


