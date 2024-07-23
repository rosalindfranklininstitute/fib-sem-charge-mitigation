from autoscript_sdb_microscope_client import SdbMicroscopeClient

from pyscanengine.scan_engine import ScanEngine
from pyscanengine.data.frame_monitor import FrameMonitor
from pyscanengine.patterns.util import convert_pattern_to_1D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image
from tifffile import imwrite

"Change microscope mode to external scan"
microscope = SdbMicroscopeClient()
microscope.connect("localhost")
microscope.imaging.set_active_view(1)
microscope.beams.electron_beam.scanning.mode.set_external()

print("Scan gen preparing...")

"Initialize QD scan generator"
scan_generator = ScanEngine(1)
scan_generator.stop_imaging(True)
scan_generator.pixel_time = 1e-7#dwell time in seconds
scan_generator.set_flyback_time(0)# flyback time in seconds
input = [1]#define input channel of QD scan generator where analog detector is connected


"Define frame size" 
#Frame size should be the same as in the scan pattern called below in: scan_array_2index = np.genfromtx...
x_size = 2048
y_size = x_size

"Number of frames for frame integration"
integration = 100

"Folder to save the data"
folder = 'D:\\Data\\2024\\nt26276-610\\Perpendicular\\2k_test\\100ns\\IL\\'

"Initialize an empty frame used for the integration afterwards"
#This integration is for reference only, without proper alignment at this point
image_int = np.zeros((y_size, x_size))

"Get the scan pattern from the text file (.xy) and convert it to 1D array"
scan_array_2index = np.genfromtxt('C:\\Users\\User\\Desktop\\ScanGenProject\\arbitrary patterns\\' + 'interleaved' + str(x_size) + 'x' + str(y_size) + '.xy', delimiter=",").astype(int)
pattern = convert_pattern_to_1D(scan_array_2index)

for k in range(0, integration):
    scan_generator.set_image_size(x_size, y_size)
    scan_generator.set_enabled_inputs(input)
    frame_monitor = FrameMonitor(x_size, y_size, input, max_queue_size=1)
    frame_monitor.register(scan_generator)

    scan_generator.scan_pattern.write_xy_array(x_size, y_size, pattern)
    scan_generator.start_table_imaging(0)

    scan_generator.stop_imaging(False)
    frame_monitor.wait_for_image(timeout=scan_generator.get_image_time()+1.0)
    imagen = frame_monitor.pop()
    imagen = imagen.get_input_data(input[0])

    "Convert to uint 8bits during acquisition"
    imagen = imagen.astype(np.int32)
    imagen = imagen + 32768
    imagen = imagen * 255 / 65535
    imagen = imagen.astype(np.uint8)

    #Integrate the frames, for reference only
    image_int = (imagen + image_int)
    
    "Save individual frames as numpy arrays"
    np.save(folder + 'IL_FI_' + str(k), imagen)
    print("Scan gen - frame ", k, " saved.")

"Save integrated frame, for reference only"
matplotlib.image.imsave(folder + 'IL_INT.tiff',image_int, cmap ='gray')


"Go back to normal microscope mode"
microscope.beams.electron_beam.scanning.mode.set_full_frame()
microscope.disconnect()

scan_generator.close()

print("Scan gen imaging success!")


"Create stack of tif images"
stack = np.empty((x_size, y_size), dtype=np.uint8)
stack = stack.reshape(1, x_size, y_size)

for i in range(0, integration):
    image_tif = np.load(folder + 'IL_FI_' + str(i) + '.npy')
    stack = np.concatenate((stack, image_tif.reshape(1, x_size, y_size)), axis =0)

stack = stack[1: integration + 1, :, :]


imwrite(folder + 'stack.tiff', stack)


