import numpy as np
from scipy.special import gamma
from scipy.spatial import KDTree
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

def count_points_in_circle(x_sensor, y_sensor, Rs, VarMaxX, VarMaxY, area_matrix):
    x_min = max(0, int(x_sensor - Rs))
    x_max = min(VarMaxX, int(x_sensor + Rs) + 1)
    y_min = max(0, int(y_sensor - Rs))
    y_max = min(VarMaxY, int(y_sensor + Rs) + 1)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            distance = np.sqrt((x - x_sensor)**2 + (y - y_sensor)**2)
            if distance <= Rs:
                area_matrix[x, y] = 1

def fitness_function(solution, num_sensor, Rs, VarMaxX, VarMaxY, ban_position):
    area = (VarMaxX + 1) * (VarMaxY + 1)
    area_matrix = np.zeros((VarMaxX+1, VarMaxY+1), dtype=int)
    node_pos = np.array(solution)
    node_pos = node_pos.reshape(-1, 2)

    for i in range(num_sensor):
        x, y = node_pos[i, :]
        count_points_in_circle(x, y, Rs, VarMaxX, VarMaxY, area_matrix)

    # for i in ban_position:
    #     area_matrix[i[0], i[1]] = 0 

    coverage_points = np.sum(area_matrix == 1)
    return 1 - (coverage_points / area)

def check_connectivity(solution, num_sensor, Rc):
    # Reshape solution into coordinates
    node_pos = np.array(solution).reshape(-1, 2)
    
    # Use KDTree to find pairs of points within Rc
    tree = KDTree(node_pos)
    pairs = tree.query_pairs(Rc)
    
    # Create a graph and add edges for each pair found
    graph = nx.Graph()
    graph.add_nodes_from(range(num_sensor))
    graph.add_edges_from(pairs)
    
    # Check if the graph is fully connected
    return nx.is_connected(graph)

def check_obstacle(x_pos, y_pos, VarMaxX, VarMaxY, obsArea):
    xj = int(x_pos)
    yj = int(y_pos)
    xj_c = xj+1
    yj_c = yj+1
    xj_t = xj-1
    yj_t = yj-1
    xj_c = np.clip(xj_c, 0, VarMaxX)
    yj_c = np.clip(yj_c, 0, VarMaxY)
    xj_t = np.clip(xj_t, 0, VarMaxX)
    yj_t = np.clip(yj_t, 0, VarMaxY)
    if (obsArea[yj,xj] == 255
        and obsArea[yj,xj_c] == 255 
        and obsArea[yj,xj_t] == 255
        and obsArea[yj_c,xj] == 255
        and obsArea[yj_t,xj] == 255):
        return True
    
    return False

###############################################################################
############################### SENSOR FUNCTION ###############################
###############################################################################

# Generate new position
def generate_new_position(x_prev, y_prev, VarMaxX, VarMaxY, Rc):
    check = True
    while check is True:
        r = np.random.uniform(Rc * 0.5, Rc)
        theta = np.random.uniform(0, 2 * np.pi)

        x_new = x_prev + r * np.cos(theta)
        y_new = y_prev + r * np.sin(theta)
        
        x_new = np.clip(x_new, 0, VarMaxX)
        y_new = np.clip(y_new, 0, VarMaxY)

        # if (check_obstacle(x_new, y_new, VarMaxX, VarMaxY, obsArea) is True):
        #     check = False
        check = False
    
    return x_new, y_new

def generate_new_solution(num_sensor, Rc, VarMaxX, VarMaxY):
    solution = np.zeros((num_sensor, 2))  # Initialize matrix num_sensor x 2
    x0, y0 = 50, 50  # Initial sensor position
    solution[0, :] = [x0, y0]  # Set initial sensor position
    # Generate positions for the rest of the sensors
    for i in range(1, num_sensor):
        x_prev, y_prev = solution[i - 1, :]
        x_new, y_new = generate_new_position(x_prev, y_prev, VarMaxX, VarMaxY, Rc)
        solution[i, :] = [x_new, y_new]
    
    return solution

def generate_new_generation(num_sensor, num_solutions, Rc, VarMaxX, VarMaxY):
    initPop = []
    for _ in range(num_solutions):
        solution = generate_new_solution(num_sensor,Rc, VarMaxX, VarMaxY)
        initPop.append(solution)

    print("Done initialize pop")

    return initPop

###############################################################################
############################### MAIN FUNCTION #################################
###############################################################################

def NOA(nPop, MaxIt,
        nNode, Rs, Rc,
        VarMin, VarMax,
        ban_position):

    dim = 2 * nNode
    lb = VarMin * np.ones(dim)
    ub = VarMax * np.ones(dim)
    # [x11, y11, x12, y12, x13, y13, ... x1n, y1n]

    bestSol = np.zeros(dim)  # Best solution so far

    bestFit = np.inf      # Best score so far
    LFit = np.full((nPop, 1), np.inf)  # Local best for each Nutcracker
    RP = np.zeros((2, dim))  # Reference points
    Convergence_curve = np.zeros(MaxIt)  # Convergence history
    
    Alpha = 0.05  # Controlling parameters
    Pa2 = 0.2
    Prb = 0.05

    pop = generate_new_generation(nNode, nPop, Rc, VarMax, VarMax)
    pop = np.array([ [element for pair in row for element in pair] for row in pop])

    Lbest = pop.copy()  # Local best position initialization
    t = 0  # Iteration counter
    
    # Evaluation
    NC_Fit = np.zeros(nPop)
    for i in range(nPop):
        # Calculate fitness value for each solution
        NC_Fit[i] = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1, ban_position)
        LFit[i] = NC_Fit[i]  # Set local best
        if NC_Fit[i] < bestFit:
            bestFit = NC_Fit[i]
            bestSol = pop[i, :].copy()
            # [x11, y11, x12, y12, x13, y13, ... x1n, y1n]
    
    # in main loop
    while t < MaxIt:
        print(f"Iteration {t}, Best Coverage: {1 - bestFit:.4f}")
        RL = 0.05 * levy(nPop, dim, 1.5)

        l = np.random.rand() * (1 - t / MaxIt)
        
        if np.random.rand() < np.random.rand():
            if (t == 0):
                a = 0
            else:
                a = (t / MaxIt) ** (2 * 1 / t)
        else:
            a = (1 - (t / MaxIt)) ** (2 * (t / MaxIt))
        
        if np.random.rand() < np.random.rand():
            mo = np.mean(pop, axis=0) # mean of 
            for i in range(nPop):
                if np.random.rand() < np.random.rand():
                    mu = np.random.rand()
                elif np.random.rand() < np.random.rand():
                    mu = np.random.randn()
                else:
                    mu = RL[0, 0]
                
                cv = np.random.randint(nPop)
                cv1 = np.random.randint(nPop)
                Pa1 = (MaxIt - t) / MaxIt

                # Exploration phase 1
                if np.random.rand() < Pa1:
                    for j in range(dim):
                        cv2 = np.random.randint(nPop)
                        r2 = np.random.rand()
                        if t < MaxIt / 2: # move global random
                            if np.random.rand() > np.random.rand():
                                temporary_pos = pop[i, j]
                                pop[i, j] = mo[j] + RL[i,j] * (pop[cv, j] - pop[cv1,j]) + mu * (np.random.rand() < 5) * (r2 * r2 * ub[j] - lb[j])
                                pop[i, j] = np.clip(pop[i, j], lb[j], ub[j])
                                if (check_connectivity(pop[i, :], nNode, Rc) == False):
                                    pop[i, j] = temporary_pos

                        else: # explore around a random solution
                            if np.random.rand() > np.random.rand():
                                temporary_pos = pop[i, j]
                                pop[i, j] = pop[cv2, j] + 0.1 * mu * (pop[cv, j] - pop[cv1, j]) + 0.1 * mu * (np.random.rand() < Alpha) * (r2 * r2 * ub[j] - lb[j])
                                pop[i, j] = np.clip(pop[i, j], lb[j], ub[j])
                                if (check_connectivity(pop[i, :], nNode, Rc) == False):
                                    pop[i, j] = temporary_pos

                # Exploitation phase 1
                else:
                    mu = np.random.rand()
                    if np.random.rand() < np.random.rand():
                        r1 = np.random.rand()
                        for j in range(dim):
                            temporary_pos = pop[i, j]
                            pop[i, j] = pop[i, j] + mu * abs(RL[i, j]) * (bestSol[j] - pop[i, j]) + 0.5 * r1 * (pop[cv, j] - pop[cv1, j])
                            pop[i, j] = np.clip(pop[i, j], lb[j], ub[j])
                            if (check_connectivity(pop[i, :], nNode, Rc) == False):
                                    pop[i, j] = temporary_pos
                                
                    elif np.random.rand() < np.random.rand():
                        for j in range(dim):
                            if np.random.rand() > np.random.rand():
                                temporary_pos = pop[i, j]
                                pop[i, j] = bestSol[j] + mu * 0.5 * (pop[cv, j] - pop[cv1, j])
                                pop[i, j] = np.clip(pop[i, j], lb[j], ub[j])
                                if (check_connectivity(pop[i, :], nNode, Rc) == False):
                                    pop[i, j] = temporary_pos                    
                                 
                    else:
                        for j in range(dim):
                            temporary_pos = pop[i, j]
                            pop[i, j] = bestSol[j] * abs(l)
                            pop[i, j] = np.clip(pop[i, j], lb[j], ub[j])
                            if (check_connectivity(pop[i, :], nNode, Rc) == False):
                                    pop[i, j] = temporary_pos

                NC_Fit[i] = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1, ban_position)
            
                # Update the local best according to Eq. (20)
                if NC_Fit[i] < LFit[i]: # Change this to > for maximization problem
                    LFit[i] = NC_Fit[i]  # Update the local best fitness
                    Lbest[i, :] = pop[i, :].copy() # Update the local best position
                else:
                    NC_Fit[i] = LFit[i]
                    pop[i, :] = Lbest[i, :]
                
                if NC_Fit[i] < bestFit: # Change this to > for maximization problem
                    bestFit = NC_Fit[i] # Update best-so-far fitness
                    bestSol = pop[i, :].copy() # Update best-so-far 
        
        else:
            # Cache-search and Recovery strategy
            ## Compute the reference points for each Nutcraker
            for i in range(nPop):
                ang = np.pi * np.random.rand()
                cv = np.random.randint(nPop)
                cv1 = np.random.randint(nPop)
                for j in range(dim):
                    for j1 in range(2):
                        if j1 == 1:
                            # Random position of 1st object around sensor 
                            if ang != np.pi / 2:
                                RP[j1, j] = pop[i, j] + a * np.cos(ang) * (pop[cv, j] - pop[cv1, j]) * 0.3
                            else:
                                RP[j1, j] = pop[i, j] + a * RP[np.random.randint(2), j] * 0.5
                        else:
                            # Compute the second reference point for the ith Nutcraker
                            if ang != np.pi / 2:
                                RP[j1, j] = pop[i, j] + a * np.cos(ang) * ((ub[j] - lb[j]) + lb[j]) * (np.random.rand() < Prb) * 0.5
                            else:
                                RP[j1, j] = pop[i, j] + a * RP[np.random.randint(2), j] * (np.random.rand() < Prb) * 0.75
                
                RP[1, :] = np.clip(RP[1, :], lb, ub)
                RP[0, :] = np.clip(RP[0, :], lb, ub)

                # Exploitation phase 2  
                if np.random.rand() < Pa2:
                    cv = np.random.randint(nPop)
                    for j in range(dim):
                        if np.random.rand() < np.random.rand():
                            if np.random.rand() > np.random.rand():
                                temporary_pos = pop[i, j]
                                pop[i, j] = pop[i, j] + (np.random.rand() * (bestSol[j] - pop[i, j]) + np.random.rand() * (RP[0, j] - pop[cv, j])) * 0.5
                                pop[i, j] = np.clip(pop[i, j], lb[j], ub[j])
                                # # Check obstacle
                                # if j % 2 == 0:
                                #     if (check_obstacle(pop[i, j], pop[i, j + 1], VarMax, VarMax, obsArea) is False):
                                #         limit_loop += 1
                                #         pop[i, j] = temporary_pos
                                #         continue
                                # else:
                                #     if (check_obstacle(pop[i, j - 1], pop[i, j], VarMax, VarMax, obsArea) is False):
                                #         limit_loop += 1
                                #         pop[i, j] = temporary_pos
                                #         continue
                                # Check connectivity
                                if (check_connectivity(pop[i, :], nNode, Rc) == False):
                                    pop[i, j] = temporary_pos
                                
                        else:
                            # cv = np.random.randint(nPop)
                            if np.random.rand() > np.random.rand(): # global search if nutcracker does not find
                                temporary_pos = pop[i, j]
                                pop[i, j] = pop[i, j] + (np.random.rand() * (bestSol[j] - pop[i, j]) + np.random.rand() * (RP[1, j] - pop[cv, j])) * 0.5
                                pop[i, j] = np.clip(pop[i, j], lb[j], ub[j])
                    
                    NC_Fit[i] = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1, ban_position)
                    
                    # Update the local best
                    if NC_Fit[i] < LFit[i]: # Change this to > for maximization problem
                        LFit[i] = NC_Fit[i]
                        Lbest[i, :] = pop[i, :].copy()
                    else:
                        NC_Fit[i] = LFit[i]
                        pop[i, :] = Lbest[i, :]
                    
                    # Update the best-so-far solution
                    if NC_Fit[i] < bestFit:
                        bestFit = NC_Fit[i] # Update best-so-far fitness
                        bestSol = pop[i, :].copy() # Update best-so-far 
                
                # Exploration stage 2: Cache-search stage
                else:
                    # Evaluation
                    NC_Fit1 = fitness_function(RP[0], nNode, Rs, VarMax + 1 , VarMax + 1, ban_position)
                    
                    # Evaluations
                    NC_Fit2 = fitness_function(RP[1], nNode, Rs, VarMax + 1, VarMax + 1, ban_position)
                    
                    # Applying Eq. (17) to trade-off between the exploration behaviors
                    if NC_Fit2 < NC_Fit1 and NC_Fit2 < NC_Fit[i]:
                        temp = RP[1, :]
                        if (check_connectivity(RP[1, :], nNode, Rc) == True):
                            NC_Fit[i] = NC_Fit2
                            pop[i, :] = temp

                    elif NC_Fit1 < NC_Fit2 and NC_Fit1 < NC_Fit[i]:
                        temp = RP[0, :]
                        if (check_connectivity(RP[0, :], nNode, Rc) == True):
                            pop[i, :] = temp
                            NC_Fit[i] = NC_Fit1
                    
                    # Update the local best
                    if NC_Fit[i] < LFit[i]:
                        LFit[i] = NC_Fit[i]
                        Lbest[i, :] = pop[i, :].copy()
                    else:
                        NC_Fit[i] = LFit[i]
                        pop[i, :] = Lbest[i, :]
                    
                    # Update the best-so-far solution
                    if NC_Fit[i] < bestFit:
                        bestFit = NC_Fit[i]
                        bestSol = pop[i, :].copy()

        t += 1
        Convergence_curve[t - 1] = bestFit
        if t >= MaxIt:
            break

    return bestFit, bestSol, Convergence_curve, t
    
