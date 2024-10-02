import numpy as np
import networkx as nx

# Generate new position
def generate_new_position(x_prev, y_prev, area_width, area_length ,communication_range):
    r = np.random.uniform(communication_range * 0.5, communication_range)
    theta = np.random.uniform(0, 2 * np.pi)
    
    x_new = x_prev + r * np.cos(theta)
    y_new = y_prev + r * np.sin(theta)
    
    x_new = np.clip(x_new, 0, area_width)
    y_new = np.clip(y_new, 0, area_length)
    
    return x_new, y_new

def generate_new_solution(num_sensor, communication_range, area_width, area_length):
    solution = np.zeros((num_sensor, 2))  # Initialize matrix num_sensor x 2
    x0, y0 = 50, 50  # Initial sensor position
    solution[0, :] = [x0, y0]  # Set initial sensor position
    # Generate positions for the rest of the sensors
    for i in range(1, num_sensor):
        random_node = i-1
        x_prev, y_prev = solution[random_node, :]
        x_new, y_new = generate_new_position(x_prev, y_prev, area_width, area_length, communication_range)
        solution[i, :] = [x_new, y_new]
    
    return solution

def generate_new_generation(num_sensor, num_solutions, communication_range, area_width, area_length):
    initPop = []
    for _ in range(num_solutions):
        solution = generate_new_solution(num_sensor,communication_range, area_width, area_length)
        initPop.append(solution)

    return initPop