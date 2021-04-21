import Image
import numpy as np
import cv2

from bresenham import bresenham

class Map():
    def __init__(self, filename, angles):
        """
        filename: path to map file
        angles:   numpy array of angles to use to produce measurements
        """
        im = Image.open(filename)
        # Flatten to 2D since grayscale:
        self.map = np.asarray(im)[:,:,0]
        self.map = (255 - self.map) / 255.0
        self.angles = angles

        self.max_range = 3.5
        self.min_range = 0.12

    def metres2pixels(self, x, y):
        """ Converts (x,y) in metres to pixel coordinates """
        if not -3 <= x <= 4.5 or not -3 <= y <= 4.5:
            raise Exception("("+str(x)+", "+str(y)+") is out of map bounds")

        x_p = int(np.around((x + 3.0) / 0.05))
        y_p = int(np.around((4.5 - y) / 0.05))
        return (x_p, y_p)

    def is_occupied(self, x, y):
        return self.map[y, x] > 0.65

    def compute_measurement(self, x, y, th):
        """ Computes a LIDAR measurements taken at (x,y) in metres on the map """
        x_start, y_start = self.metres2pixels(x, y)  # Convert to pixel coordinates
        angles = self.angles + th                   # Adjust for yaw angle

        measurements = []
        for angle in angles:
            # Maximum LIDAR measurement range
            x_end = int(np.around(x_start + (self.max_range / 0.05) * np.cos(angle)))
            y_end = int(np.around(y_start + (self.max_range / 0.05) * np.sin(angle)))
            # Points on a line at an angle from (x,y) outwards for max_range metres
            line = list(bresenham(x_start, y_start, x_end, y_end))

            # Check each point along line for being occupied
            object_point = None
            for (x_p, y_p) in line:
                if not 0 <= x_p < 150 or not 0 <= y_p < 150:
                    # Out of image bounds
                    break
                elif self.is_occupied(x_p, y_p):
                    object_point = np.array([x_p, y_p])
                    break
            
            if object_point is None:
                # Not occupied so measurement is maximum range
                measurements.append(self.max_range)
            else:
                pixels_to_object = np.linalg.norm(object_point - np.array([x_start, y_start]))
                dist_to_object = pixels_to_object * 0.05
                # Measurement is maximum of min_range and object point after converting to metres
                measurements.append(max(self.min_range, dist_to_object))

        return measurements


