import numpy as np
from scipy.special import gamma
import random
import networkx as nx

###############################################################################
############################## GENERAL FUNCTION ###############################
###############################################################################

def levy(n, m, beta):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)
    sigma_u = (num / den)**(1 / beta)
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    z = u / (np.abs(v)**(1 / beta))
    
    return z

def count_points_in_circle(x_sensor, y_sensor, sensing_range, area_width, area_length, area_matrix):
    x_min = max(0, int(x_sensor - sensing_range))
    x_max = min(area_width, int(x_sensor + sensing_range) + 1)
    y_min = max(0, int(y_sensor - sensing_range))
    y_max = min(area_length, int(y_sensor + sensing_range) + 1)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            distance = np.sqrt((x - x_sensor)**2 + (y - y_sensor)**2)
            if distance <= sensing_range:
                area_matrix[x, y] = 1

def fitness_function(solution, num_sensor, sensing_range, area_width, area_length, area, area_matrix, obstacle_map):
    area_matrix[:] = 0
    node_pos = np.array(solution)
    node_pos = node_pos.reshape(-1, 2)
    obstacle_points = np.sum(obstacle_map == 2)

    for i in range(num_sensor):
        x, y = node_pos[i, :]
        count_points_in_circle(x, y, sensing_range, area_width, area_length, area_matrix)\
        
    area_matrix = area_matrix + obstacle_map
    
    coverage_points = np.sum(area_matrix == 1)
    return 1 - (coverage_points / (area - obstacle_points))

def check_connectivity(solution, num_sensor, communication_range):
    adjacency_matrix = np.zeros((num_sensor, num_sensor))
    node_pos = np.array(solution)
    node_pos = node_pos.reshape(-1, 2)
    
    for i in range(num_sensor):
        for j in range(i + 1, num_sensor):
            x_i, y_i = node_pos[i, :]
            x_j, y_j = node_pos[j, :]
            distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            if distance <= communication_range:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    graph = nx.Graph(adjacency_matrix)
    connected_components = list(nx.connected_components(graph))
    # print(len(connected_components) == 1)
    return len(connected_components) == 1

###############################################################################
############################### SENSOR FUNCTION ###############################
###############################################################################

# Generate new position
def generate_new_position(x_prev, y_prev, area_width, area_length ,communication_range, obstacle_radius):
    center_map = area_width // 2
    while True:
        r = np.random.uniform(communication_range * 0.5, communication_range)
        theta = np.random.uniform(0, 2 * np.pi)

        x_new = x_prev + r * np.cos(theta)
        y_new = y_prev + r * np.sin(theta)
        
        x_new = np.clip(x_new, 0, area_width)
        y_new = np.clip(y_new, 0, area_length)

        distance_to_center = np.sqrt((x_new - center_map)**2 + (y_new - center_map)**2)

        if (distance_to_center > obstacle_radius):
            break
    
    return x_new, y_new

def generate_new_solution(num_sensor, communication_range, area_width, area_length, obstacle_radius):
    solution = np.zeros((num_sensor, 2))  # Initialize matrix num_sensor x 2
    x0, y0 = 20, 20  # Initial sensor position
    solution[0, :] = [x0, y0]  # Set initial sensor position
    # Generate positions for the rest of the sensors
    for i in range(1, num_sensor):
        random_node = i-1
        x_prev, y_prev = solution[random_node, :]
        x_new, y_new = generate_new_position(x_prev, y_prev, area_width, area_length, communication_range, obstacle_radius)
        solution[i, :] = [x_new, y_new]
    
    return solution

def generate_new_generation(num_sensor, num_solutions, communication_range, area_width, area_length, obstacle_radius):
    initPop = []
    for _ in range(num_solutions):
        solution = generate_new_solution(num_sensor,communication_range, area_width, area_length, obstacle_radius)
        initPop.append(solution)

    print("Done initialize population")

    return initPop

# change node position if it's inside obstacle
def adjust_sensor_position(x_pos, y_pos, max_length, obstacle_radius):
    center = max_length // 2

    vector_x = x_pos - center
    vector_y = y_pos - center
    distance = np.sqrt(vector_x ** 2 + vector_y ** 2)

    if distance == 0:
        angle = random.uniform(0, 2 * np.pi)
        new_x_pos = center + (obstacle_radius + 0.1) * np.cos(angle)
        new_y_pos = center + (obstacle_radius + 0.1) * np.sin(angle)

    else:
        scale = (obstacle_radius + 10) / distance
        new_x_pos = center + vector_x * scale
        new_y_pos = center + vector_y * scale
        
    # Kiểm tra nếu tọa độ mới nằm trong phạm vi của ma trận
    if 0 <= new_x_pos < max_length and 0 <= new_y_pos < max_length:
        return new_x_pos, new_y_pos
    else:
        print("Tọa độ mới nằm ngoài phạm vi của ma trận.")
        return x_pos, y_pos

###############################################################################
############################### MAIN FUNCTION #################################
###############################################################################

def NOA(SearchAgents_no, Max_iter,
        num_sensors, sensing_range, communication_range,
        min_length, max_length,
        obstacle_map, obstacle_radius):

    dim = 2 * num_sensors
    lb = min_length * np.ones(dim)
    ub = max_length * np.ones(dim)
    center_map = max_length // 2
    # [x11, y11, x12, y12, x13, y13, ... x1n, y1n]

    area = max_length * max_length
    area_matrix = np.zeros((max_length, max_length))

    best_solution = np.zeros(dim)  # Best solution so far

    best_fitness = np.inf      # Best score so far
    LFit = np.full((SearchAgents_no, 1), np.inf)  # Local best for each Nutcracker
    RP = np.zeros((2, dim))  # Reference points
    Convergence_curve = np.zeros(Max_iter)  # Convergence history
    
    Alpha = 0.05  # Controlling parameters
    Pa2 = 0.2
    Prb = 0.2

    population = generate_new_generation(num_sensors, SearchAgents_no, communication_range, max_length, max_length, obstacle_radius)
    population = np.array([ [element for pair in row for element in pair] for row in population])

    # Output: matrix [solutions * dim]
    # [
    #   [x11, y11, x12, y12, x13, y13, ... x1n, y1n],
    #   [x21, y21, x22, y22, x23, y23, ... x2n, y2n],
    #   ....
    #   [xm1, ym1, xm2, ym2, xm3, ym3, ... xmn, ymn],
    # ]
    # n = sensors, m = solutions = agents = nutcrackers

    Lbest = population.copy()  # Local best position initialization
    t = 0  # Iteration counter
    
    # Evaluation
    NC_Fit = np.zeros(SearchAgents_no)
    for i in range(SearchAgents_no):
        # Calculate fitness value for each solution
        NC_Fit[i] = fitness_function(population[i, :], num_sensors, sensing_range, max_length, max_length, area, area_matrix, obstacle_map)
        LFit[i] = NC_Fit[i]  # Set local best
        if NC_Fit[i] > best_fitness:
            best_fitness = NC_Fit[i]
            best_solution = population[i, :].copy()
            # [x11, y11, x12, y12, x13, y13, ... x1n, y1n]
    
    # in main loop
    while t < Max_iter:
        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)
        # Output: matrix [solutions * dim]
        # [
        #   [x11, y11, x12, y12, x13, y13, ... x1n, y1n],
        #   [x21, y21, x22, y22, x23, y23, ... x2n, y2n],
        #   ....
        #   [xm1, ym1, xm2, ym2, xm3, ym3, ... xmn, ymn],
        # ]

        l = np.random.rand() * (1 - t / Max_iter)
        
        if np.random.rand() < np.random.rand():
            if (t == 0):
                a = 0
            else:
                a = (t / Max_iter) ** (2 * 1 / t)
        else:
            a = (1 - (t / Max_iter)) ** (2 * (t / Max_iter))
        
        if np.random.rand() < np.random.rand():
            mo = np.mean(population, axis=0) # mean of 
            # Output 
            # [[x_tb_1, y_tb_1], [x_tb_2, y_tb_2], [x_tb_3, y_tb_3], ... [x_tb_n, y_tb_n]]
            for i in range(SearchAgents_no):
                # print(f"Iteration {t}, solution {SearchAgents_no}")
                if np.random.rand() < np.random.rand():
                    mu = np.random.rand()
                elif np.random.rand() < np.random.rand():
                    mu = np.random.randn()
                else:
                    mu = RL[0, 0]
                
                cv = np.random.randint(SearchAgents_no)
                cv1 = np.random.randint(SearchAgents_no)
                Pa1 = (Max_iter - t) / Max_iter

                # Exploration phase 1
                if np.random.rand() < Pa1:
                    for j in range(dim):
                        limit_loop = 0
                        while limit_loop < 25:
                            cv2 = np.random.randint(SearchAgents_no)
                            r2 = np.random.rand()
                            if t < Max_iter / 2: # move global random
                                if np.random.rand() > np.random.rand():
                                    temporary_pos = population[i, j]
                                    population[i, j] = mo[j] + RL[i,j] * (population[cv, j] - population[cv1,j]) + mu * (np.random.rand() < 5) * (r2 * r2 * ub[j] - lb[j])
                                    population[i, j] = np.clip(population[i, j], lb[j], ub[j])
                                    # Check obstacle
                                    if (j % 2 == 0):
                                        distance_to_center = np.sqrt((population[i, j] - center_map)**2 + (population[i, j + 1] - center_map)**2)
                                        if (distance_to_center <= obstacle_radius):
                                            population[i, j], population[i, j + 1] = adjust_sensor_position(population[i, j], population[i, j + 1], max_length, obstacle_radius)
                                    else:
                                        distance_to_center = np.sqrt((population[i, j - 1] - center_map)**2 + (population[i, j] - center_map)**2)
                                        if (distance_to_center <= obstacle_radius):
                                            population[i, j - 1], population[i, j] = adjust_sensor_position(population[i, j - 1], population[i, j], max_length, obstacle_radius)
                                    # Check connectivity
                                    if (check_connectivity(population[i, :], num_sensors, communication_range) == False):
                                        population[i, j] = temporary_pos
                                        limit_loop += 1
                                        break
                                    else:
                                        break

                            else: # explore around a random solution
                                if np.random.rand() > np.random.rand():
                                    temporary_pos = population[i, j]
                                    population[i, j] = population[cv2, j] + mu * (population[cv, j] - population[cv1, j]) + mu * (np.random.rand() < Alpha) * (r2 * r2 * ub[j] - lb[j])
                                    population[i, j] = np.clip(population[i, j], lb[j], ub[j])
                                    # Check obstacle
                                    if (j % 2 == 0):
                                        distance_to_center = np.sqrt((population[i, j] - center_map)**2 + (population[i, j + 1] - center_map)**2)
                                        if (distance_to_center <= obstacle_radius):
                                            population[i, j], population[i, j + 1] = adjust_sensor_position(population[i, j], population[i, j + 1], max_length, obstacle_radius)
                                    else:
                                        distance_to_center = np.sqrt((population[i, j - 1] - center_map)**2 + (population[i, j] - center_map)**2)
                                        if (distance_to_center <= obstacle_radius):
                                            population[i, j - 1], population[i, j] = adjust_sensor_position(population[i, j - 1], population[i, j], max_length, obstacle_radius)
                                    # Check connectivity
                                    if (check_connectivity(population[i, :], num_sensors, communication_range) == False):
                                        population[i, j] = temporary_pos
                                        limit_loop += 1
                                        break
                                    else:
                                        break

                # Exploitation phase 1
                else:
                    mu = np.random.rand()
                    if np.random.rand() < np.random.rand():
                        r1 = np.random.rand()
                        for j in range(dim):
                            temporary_pos = population[i, j]
                            population[i, j] = population[i, j] + mu * abs(RL[i, j]) * (best_solution[j] - population[i, j]) + r1 * (population[cv, j] - population[cv1, j])
                            population[i, j] = np.clip(population[i, j], lb[j], ub[j])
                            # Check obstacle
                            if (j % 2 == 0):
                                distance_to_center = np.sqrt((population[i, j] - center_map)**2 + (population[i, j + 1] - center_map)**2)
                                if (distance_to_center <= obstacle_radius):
                                    population[i, j], population[i, j + 1] = adjust_sensor_position(population[i, j], population[i, j + 1], max_length, obstacle_radius)
                            else:
                                distance_to_center = np.sqrt((population[i, j - 1] - center_map)**2 + (population[i, j] - center_map)**2)
                                if (distance_to_center <= obstacle_radius):
                                    population[i, j - 1], population[i, j] = adjust_sensor_position(population[i, j - 1], population[i, j], max_length, obstacle_radius)
                            # Check connectivity
                            if (check_connectivity(population[i, :], num_sensors, communication_range) == False):
                                population[i, j] = temporary_pos
                                
                    elif np.random.rand() < np.random.rand():
                        for j in range(dim):
                            if np.random.rand() > np.random.rand():
                                temporary_pos = population[i, j]
                                population[i, j] = best_solution[j] + mu * (population[cv, j] - population[cv1, j])
                                population[i, j] = np.clip(population[i, j], lb[j], ub[j])
                                # Check obstacle
                                if (j % 2 == 0):
                                    distance_to_center = np.sqrt((population[i, j] - center_map)**2 + (population[i, j + 1] - center_map)**2)
                                    if (distance_to_center <= obstacle_radius):
                                        population[i, j], population[i, j + 1] = adjust_sensor_position(population[i, j], population[i, j + 1], max_length, obstacle_radius)
                                else:
                                    distance_to_center = np.sqrt((population[i, j - 1] - center_map)**2 + (population[i, j] - center_map)**2)
                                    if (distance_to_center <= obstacle_radius):
                                        population[i, j - 1], population[i, j] = adjust_sensor_position(population[i, j - 1], population[i, j], max_length, obstacle_radius)
                                # Check connectivity
                                if (check_connectivity(population[i, :], num_sensors, communication_range) == False):
                                    population[i, j] = temporary_pos                        
                                 
                    else:
                        for j in range(dim):
                            temporary_pos = population[i, j]
                            population[i, j] = best_solution[j] * abs(l)
                            population[i, j] = np.clip(population[i, j], lb[j], ub[j])
                            # Check obstacle
                            if (j % 2 == 0):
                                distance_to_center = np.sqrt((population[i, j] - center_map)**2 + (population[i, j + 1] - center_map)**2)
                                if (distance_to_center <= obstacle_radius):
                                    population[i, j], population[i, j + 1] = adjust_sensor_position(population[i, j], population[i, j + 1], max_length, obstacle_radius)
                            else:
                                distance_to_center = np.sqrt((population[i, j - 1] - center_map)**2 + (population[i, j] - center_map)**2)
                                if (distance_to_center <= obstacle_radius):
                                    population[i, j - 1], population[i, j] = adjust_sensor_position(population[i, j - 1], population[i, j], max_length, obstacle_radius)
                            # Check connectivity
                            if (check_connectivity(population[i, :], num_sensors, communication_range) == False):
                                population[i, j] = temporary_pos
                
                # Ensure the search agents stay within the boundaries
                # Move -> clip -> check connectivity
                # population[i, :] = np.clip(population[i, :], lb, ub)

                NC_Fit[i] = fitness_function(population[i, :], num_sensors, sensing_range, max_length, max_length, area, area_matrix, obstacle_map)
            
                # Update the local best according to Eq. (20)
                if NC_Fit[i] < LFit[i]: # Change this to > for maximization problem
                    LFit[i] = NC_Fit[i]  # Update the local best fitness
                    Lbest[i, :] = population[i, :].copy() # Update the local best position
                else:
                    NC_Fit[i] = LFit[i]
                    population[i, :] = Lbest[i, :]
                
                if NC_Fit[i] < best_fitness: # Change this to > for maximization problem
                    best_fitness = NC_Fit[i] # Update best-so-far fitness
                    best_solution = population[i, :].copy() # Update best-so-far 
                
                t += 1
                if t >= Max_iter:
                    break
                
                Convergence_curve[t - 1] = best_fitness
        
        else:
            # Cache-search and Recovery strategy
            ## Compute the reference points for each Nutcraker
            for i in range(SearchAgents_no):
                # print(f"Iteration {t}, solution {SearchAgents_no}")
                ang = np.pi * np.random.rand()
                cv = np.random.randint(SearchAgents_no)
                cv1 = np.random.randint(SearchAgents_no)
                for j in range(dim):
                    for j1 in range(2):
                        if j1 == 1:
                            # Random position of 1st object around sensor 
                            if ang != np.pi / 2:
                                RP[j1, j] = population[i, j] + a * np.cos(ang) * (population[cv, j] - population[cv1, j])
                            else:
                                RP[j1, j] = population[i, j] + a * np.cos(ang) * (population[cv, j] - population[cv1, j]) + a * RP[np.random.randint(2), j]
                        else:
                            # Compute the second reference point for the ith Nutcraker
                            if ang != np.pi / 2:
                                RP[j1, j] = population[i, j] + a * np.cos(ang) * ((ub[j] - lb[j]) + lb[j]) * (np.random.rand() < Prb)
                            else:
                                RP[j1, j] = population[i, j] + a * np.cos(ang) * ((ub[j] - lb[j]) * np.random.rand() + lb[j]) + a * RP[np.random.randint(2), j] * (np.random.rand() < Prb)
                
                RP[1, :] = np.clip(RP[1, :], lb, ub)
                RP[0, :] = np.clip(RP[0, :], lb, ub)
                
                # Exploitation phase 2  
                if np.random.rand() < Pa2:
                    cv = np.random.randint(SearchAgents_no)
                    if np.random.rand() < np.random.rand():
                        for j in range(dim):
                            limit_loop = 0
                            while limit_loop < 25:
                                # cv = np.random.randint(SearchAgents_no)
                                if np.random.rand() > np.random.rand():
                                    temporary_pos = population[i, j]
                                    population[i, j] = population[i, j] + np.random.rand() * (best_solution[j] - population[i, j]) + np.random.rand() * (RP[0, j] - population[cv, j])
                                    population[i, j] = np.clip(population[i, j], lb[j], ub[j])
                                    # Check obstacle
                                    if (j % 2 == 0):
                                        distance_to_center = np.sqrt((population[i, j] - center_map)**2 + (population[i, j + 1] - center_map)**2)
                                        if (distance_to_center <= obstacle_radius):
                                            population[i, j], population[i, j + 1] = adjust_sensor_position(population[i, j], population[i, j + 1], max_length, obstacle_radius)
                                    else:
                                        distance_to_center = np.sqrt((population[i, j - 1] - center_map)**2 + (population[i, j] - center_map)**2)
                                        if (distance_to_center <= obstacle_radius):
                                            population[i, j - 1], population[i, j] = adjust_sensor_position(population[i, j - 1], population[i, j], max_length, obstacle_radius)
                                    # Check connectivity
                                    if (check_connectivity(population[i, :], num_sensors, communication_range) == False):
                                        population[i, j] = temporary_pos
                                        limit_loop += 1
                                        break
                                    else:
                                        break

                    else:
                        for j in range(dim):
                            limit_loop = 0
                            while limit_loop < 25:
                                cv = np.random.randint(SearchAgents_no)
                                if np.random.rand() > np.random.rand(): # global search if nutcracker does not find
                                    temporary_pos = population[i, j]
                                    population[i, j] = population[i, j] + np.random.rand() * (best_solution[j] - population[i, j]) + np.random.rand() * (RP[1, j] - population[cv, j])
                                    population[i, j] = np.clip(population[i, j], lb[j], ub[j])
                                    # Check obstacle
                                    if (j % 2 == 0):
                                        distance_to_center = np.sqrt((population[i, j] - center_map)**2 + (population[i, j + 1] - center_map)**2)
                                        if (distance_to_center <= obstacle_radius):
                                            population[i, j], population[i, j + 1] = adjust_sensor_position(population[i, j], population[i, j + 1], max_length, obstacle_radius)
                                    else:
                                        distance_to_center = np.sqrt((population[i, j - 1] - center_map)**2 + (population[i, j] - center_map)**2)
                                        if (distance_to_center <= obstacle_radius):
                                            population[i, j - 1], population[i, j] = adjust_sensor_position(population[i, j - 1], population[i, j], max_length, obstacle_radius)
                                    # Check connectivity
                                    if (check_connectivity(population[i, :], num_sensors, communication_range) == False):
                                        population[i, j] = temporary_pos
                                        limit_loop += 1
                                        break
                                    else:
                                        break
                    
                    NC_Fit[i] = fitness_function(population[i, :], num_sensors, sensing_range, max_length, max_length, area, area_matrix, obstacle_map)
                    
                    # Update the local best
                    if NC_Fit[i] < LFit[i]: # Change this to > for maximization problem
                        LFit[i] = NC_Fit[i]
                        Lbest[i, :] = population[i, :].copy()
                    else:
                        NC_Fit[i] = LFit[i]
                        population[i, :] = Lbest[i, :]
                    
                    # Update the best-so-far solution
                    if NC_Fit[i] < best_fitness: # Change this to > for maximization problem
                        best_fitness = NC_Fit[i] # Update best-so-far fitness
                        best_solution = population[i, :].copy() # Update best-so-far 
                    
                    t += 1
                    if t >= Max_iter:
                        break
                
                # Exploration stage 2: Cache-search stage
                else:
                    # Evaluation
                    NC_Fit1 = fitness_function(RP[0], num_sensors, sensing_range, max_length, max_length, area, area_matrix, obstacle_map)
                    
                    t += 1
                    if t >= Max_iter:
                        break
                    
                    # Evaluations
                    NC_Fit2 = fitness_function(RP[1], num_sensors, sensing_range, max_length, max_length, area, area_matrix, obstacle_map)
                    
                    # Applying Eq. (17) to trade-off between the exploration behaviors
                    if NC_Fit2 < NC_Fit1 and NC_Fit2 < NC_Fit[i]:
                        temp = RP[1, :]
                        for j in range(num_sensors):
                            distance_to_center = np.sqrt((temp[2*j] - center_map)**2 + (temp[2*j + 1] - center_map)**2)
                            if (distance_to_center <= obstacle_radius):
                                temp[2*j], temp[2*j + 1] = adjust_sensor_position(temp[2*j], temp[2*j + 1], max_length, obstacle_radius)

                        if (check_connectivity(RP[1, :], num_sensors, communication_range) == True):
                            NC_Fit[i] = NC_Fit2
                            population[i, :] = temp

                    elif NC_Fit1 < NC_Fit2 and NC_Fit1 < NC_Fit[i]:
                        temp = RP[0, :]
                        for j in range(num_sensors):
                            distance_to_center = np.sqrt((temp[2*j] - center_map)**2 + (temp[2*j + 1] - center_map)**2)
                            if (distance_to_center <= obstacle_radius):
                                temp[2*j], temp[2*j + 1] = adjust_sensor_position(temp[2*j], temp[2*j + 1], max_length, obstacle_radius)

                        if (check_connectivity(RP[0, :], num_sensors, communication_range) == True):
                            population[i, :] = temp
                            NC_Fit[i] = NC_Fit1
                    
                    # Update the local best
                    if NC_Fit[i] < LFit[i]:
                        LFit[i] = NC_Fit[i]
                        Lbest[i, :] = population[i, :].copy()
                    else:
                        NC_Fit[i] = LFit[i]
                        population[i, :] = Lbest[i, :]
                    
                    # Update the best-so-far solution
                    if NC_Fit[i] < best_fitness:
                        best_fitness = NC_Fit[i]
                        best_solution = population[i, :].copy()

                    t += 1
                    if t >= Max_iter:
                        break

        print(f"Iteration {t}, Best Coverage: {1 - best_fitness:.4f}")
    
    return best_fitness, best_solution, Convergence_curve, t
    
