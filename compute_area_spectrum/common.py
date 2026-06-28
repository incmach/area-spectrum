import numpy as np
import math

def get_zero_areas_count(spectrum):
    if len(spectrum) == 0:
        return 0
    tail_sum = sum(spectrum[1:])
    total_points_cube = spectrum[0] + tail_sum
    #TODO make look exact
    total_points_aprx = int(np.round(total_points**(1/3)))
    return math.comb(total_points_aprx, 3) - tail_sum
