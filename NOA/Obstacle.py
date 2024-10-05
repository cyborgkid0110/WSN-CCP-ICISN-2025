import numpy as np

def create_rectangle(x_start, x_end, y_start, y_end, max_length):
    if x_start < 0 or x_end > max_length or y_start < 0 or y_end > max_length:
        return
    
    if x_start >= x_end or y_start >= y_end:
        return
    
    matrix = np.zeros((max_length, max_length), dtype=int)
    matrix[x_start:x_end, y_start:y_end] = 1

    return matrix

    # for row in matrix:
    #     print(' '.join(map(str, row)))

def create_circle(obstacle_radius, max_length):
    matrix = np.zeros((max_length, max_length), dtype=int)
    center = max_length // 2

    for i in range(max_length):
        for j in range(max_length):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            
            if distance <= obstacle_radius:
                matrix[i, j] = 2

    return matrix

    # for row in matrix:
    #     print(' '.join(map(str, row)))


