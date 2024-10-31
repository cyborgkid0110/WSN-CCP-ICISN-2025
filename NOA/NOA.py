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

def fitness_function(sensor_nodes, nNode, Rs, VarMaxX, VarMaxY):
    node_pos = np.array(sensor_nodes).reshape(-1, 2)
    M = (VarMaxX + 1) * (VarMaxY + 1)
    Rss = Rs ** 2 
    matrix_c = np.zeros((VarMaxX + 1, VarMaxY + 1), dtype=int)
    grid_x, grid_y = np.meshgrid(np.arange(VarMaxX + 1), np.arange(VarMaxY + 1), indexing='ij')
    for i in range(nNode):
        sensor_x, sensor_y = node_pos[i, :]
        distances = (grid_x - sensor_x) ** 2 + (grid_y - sensor_y) ** 2
        matrix_c[distances <= Rss] = 1
    coverage_ratio = np.sum(matrix_c) / (M)
    return 1 - round(coverage_ratio, 4)


def check_connectivity(solution, nNode, Rc):
    # Reshape solution into coordinates
    node_pos = np.array(solution).reshape(-1, 2)
    
    # Use KDTree to find pairs of points within Rc
    tree = KDTree(node_pos)
    pairs = tree.query_pairs(Rc)
    
    # Create a graph and add edges for each pair found
    graph = nx.Graph()
    graph.add_nodes_from(range(nNode))
    graph.add_edges_from(pairs)
    
    # Check if the graph is fully connected
    return nx.is_connected(graph)

###############################################################################
############################### SENSOR FUNCTION ###############################
###############################################################################

# Generate new position
def generate_new_position(x_prev, y_prev, VarMaxX, VarMaxY, Rc):
    r = random.uniform(Rc * 0.5, Rc)
    theta = random.uniform(0, 2 * np.pi)

    x_new = x_prev + r * np.cos(theta)
    y_new = y_prev + r * np.sin(theta)
    
    x_new = np.clip(x_new, 0, VarMaxX)
    y_new = np.clip(y_new, 0, VarMaxY)
    
    return x_new, y_new

def generate_new_solution(nNode, Rc, VarMaxX, VarMaxY):
    solution = np.zeros((nNode, 2))  # Initialize matrix nNode x 2
    x0, y0 = 50, 50  # Initial sensor position
    solution[0, :] = [x0, y0]  # Set initial sensor position
    # Generate positions for the rest of the sensors
    for i in range(1, nNode):
        if (i == 1):
            prevNode = 0
        else:
            prevNode = random.randint(0, i - 1)
        x_prev, y_prev = solution[prevNode, :]
        x_new, y_new = generate_new_position(x_prev, y_prev, VarMaxX, VarMaxY, Rc)
        solution[i, :] = [x_new, y_new]
    
    return solution

def generate_new_generation(nNode, nPop, Rc, VarMaxX, VarMaxY):
    initPop = []
    for _ in range(nPop):
        solution = generate_new_solution(nNode,Rc, VarMaxX, VarMaxY)
        initPop.append(solution)
    print("Done initialize pop")
    return initPop

###############################################################################
############################### MAIN FUNCTION #################################
###############################################################################

def NOA(nPop, MaxIt,
        nNode, Rs, Rc,
        VarMin, VarMax):

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
    Prb = 0.2

    pop = generate_new_generation(nNode, nPop, Rc, VarMax, VarMax)
    pop = np.array([ [element for pair in row for element in pair] for row in pop])

    Lbest = pop.copy()  # Local best position initialization
    t = 0  # Iteration counter
    
    # Evaluation
    NC_Fit = np.zeros(nPop)
    for i in range(nPop):
        # Calculate fitness value for each solution
        NC_Fit[i] = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
        LFit[i] = NC_Fit[i]  # Set local best
        if NC_Fit[i] < bestFit:
            bestFit = NC_Fit[i]
            bestSol = pop[i, :].copy()
            # [x11, y11, x12, y12, x13, y13, ... x1n, y1n]
    
    # in main loop
    while t < MaxIt:
        case=0.0
        RL = 0.05 * levy(nPop, dim, 1.5)

        l = random.random() * (1 - t / MaxIt)
        
        if random.random() < random.random():
            if (t == 0):
                a = 0
            else:
                a = (t / MaxIt) ** (2 * 1 / t)
        else:
            a = (1 - (t / MaxIt)) ** (2 * (t / MaxIt))
        
        # if random.random() < 0.75:
        if random.random() < random.random():
            mo = np.mean(pop, axis=0) # mean of 
            for i in range(nPop):
                if random.random() < random.random():
                    mu = random.random()
                elif random.random() < random.random():
                    mu = random.gauss(0, 1)
                else:
                    mu = RL[0, 0]
                
                cv = random.randint(0, nPop - 1)
                cv1 = random.randint(0, nPop - 1)
                Pa1 = (MaxIt - t) / MaxIt

                currentFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)

                # Exploration phase 1
                if random.random() < Pa1:
                    for j in range(dim):
                        cv2 = random.randint(0, nPop - 1)
                        r2 = random.random()
                        if t < MaxIt / 2: # move global random
                        # if random.random() > random.random():
                            # if random.random() > random.random():
                            temporary_pos = pop[i, j]
                            pop[i, j] = mo[j] + RL[i,j] * (pop[cv, j] - pop[cv1,j])
                            pop[i, j] = max(lb[j], min(pop[i, j], ub[j]))
                            newFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                            if (check_connectivity(pop[i, :], nNode, Rc) == False or newFit > currentFit):
                                pop[i, j] = temporary_pos
                            else:
                                currentFit = newFit
                            case=1.1
                            
                        else: # explore around a random solution
                            # if random.random() > random.random():
                            temporary_pos = pop[i, j]
                            pop[i, j] = pop[cv2, j] + 0.1 * mu * (pop[cv, j] - pop[cv1, j]) + 0.1 * mu * (random.random() < Alpha) * (r2 * r2 * ub[j] - lb[j])
                            pop[i, j] = max(lb[j], min(pop[i, j], ub[j]))
                            newFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                            if (check_connectivity(pop[i, :], nNode, Rc) == False or newFit > currentFit):
                                pop[i, j] = temporary_pos
                            else:
                                currentFit = newFit
                            case=1.2

                # Exploitation phase 1
                else:
                    mu = random.random()
                    if random.random() < random.random():
                        case=2.1
                        r1 = random.random()
                        for j in range(dim):
                            temporary_pos = pop[i, j]
                            pop[i, j] = pop[i, j] + mu * abs(RL[i, j]) * (bestSol[j] - pop[i, j]) + 0.5 * r1 * (pop[cv, j] - pop[cv1, j])
                            pop[i, j] = max(lb[j], min(pop[i, j], ub[j]))
                            newFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                            if (check_connectivity(pop[i, :], nNode, Rc) == False or newFit > currentFit):
                                pop[i, j] = temporary_pos
                            else:
                                currentFit = newFit
                                
                    elif random.random() < random.random():
                        case=2.2  
                        for j in range(dim):
                            # if random.random() > random.random():
                            temporary_pos = pop[i, j]
                            pop[i, j] = bestSol[j] + mu * 0.5 * (pop[cv, j] - pop[cv1, j])
                            pop[i, j] = max(lb[j], min(pop[i, j], ub[j]))
                            newFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                            if (check_connectivity(pop[i, :], nNode, Rc) == False or newFit > currentFit):
                                pop[i, j] = temporary_pos
                            else:
                                currentFit = newFit             
                                 
                    else:
                        case=2.3
                        for j in range(dim):
                            # currentFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                            temporary_pos = pop[i, j]
                            pop[i, j] = bestSol[j] * abs(l)
                            pop[i, j] = max(lb[j], min(pop[i, j], ub[j]))
                            newFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                            if (check_connectivity(pop[i, :], nNode, Rc) == False or newFit > currentFit):
                                pop[i, j] = temporary_pos
                            else:
                                currentFit = newFit

                NC_Fit[i] = currentFit
            
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
            skipSol = False
            # Cache-search and Recovery strategy
            ## Compute the reference points for each Nutcraker
            for i in range(nPop):
                currentFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                ang = np.pi * random.random()
                cv = random.randint(0, nPop - 1)
                cv1 = random.randint(0, nPop - 1)
                for j in range(dim):
                    for j1 in range(2):
                        if j1 == 1:
                            # Random position of 1st object around sensor 
                            if ang != np.pi / 2:
                                RP[j1, j] = pop[i, j] + a * np.cos(ang) * (pop[cv, j] - pop[cv1, j]) * 0.3   
                            else:
                                RP[j1, j] = pop[i, j] + a * RP[random.randint(0, 1), j] * 0.5
                        else:
                            # Compute the second reference point for the ith Nutcraker
                            if ang != np.pi / 2:
                                RP[j1, j] = pop[i, j] + a * np.cos(ang) * ((ub[j] - lb[j]) * random.random() + lb[j]) * (random.random() < Prb) * 0.5
                            else:
                                RP[j1, j] = pop[i, j] + a * RP[random.randint(0, 1), j] * (random.random() < Prb) * 0.75
                
                RP[1, :] = np.clip(RP[1, :], lb, ub)
                RP[0, :] = np.clip(RP[0, :], lb, ub)

                # Exploitation phase 2  
                if random.random() < Pa2:
                    cv = random.randint(0, nPop - 1)
                    for j in range(dim):
                        if random.random() < random.random():
                            # if random.random() > random.random():
                            temporary_pos = pop[i, j]
                            pop[i, j] = pop[i, j] + (random.random() * (bestSol[j] - pop[i, j]) + random.random() * (RP[0, j] - pop[cv, j]))
                            pop[i, j] = max(lb[j], min(pop[i, j], ub[j]))
                            newFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                            if (check_connectivity(pop[i, :], nNode, Rc) == False or newFit > currentFit):
                                pop[i, j] = temporary_pos
                            else:
                                currentFit = newFit
                            case=3.1
                                
                        else:
                            # if random.random() > random.random(): # global search if nutcracker does not find
                            temporary_pos = pop[i, j]
                            pop[i, j] = pop[i, j] + (random.random() * (bestSol[j] - pop[i, j]) + random.random() * (RP[1, j] - pop[cv, j]))
                            pop[i, j] = max(lb[j], min(pop[i, j], ub[j]))
                            newFit = fitness_function(pop[i, :], nNode, Rs, VarMax + 1, VarMax + 1)
                            if (check_connectivity(pop[i, :], nNode, Rc) == False or newFit > currentFit):
                                pop[i, j] = temporary_pos
                            else:
                                currentFit = newFit
                            case=3.2
                    
                    NC_Fit[i] = currentFit
                    
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
                    NC_Fit1 = fitness_function(RP[0], nNode, Rs, VarMax + 1 , VarMax + 1)
                    
                    # Evaluations
                    NC_Fit2 = fitness_function(RP[1], nNode, Rs, VarMax + 1, VarMax + 1)
                    
                    # Applying Eq. (17) to trade-off between the exploration behaviors
                    if NC_Fit2 < NC_Fit1 and NC_Fit2 < NC_Fit[i]:
                        temp = RP[1, :]
                        if (check_connectivity(RP[1, :], nNode, Rc) == True):
                            NC_Fit[i] = NC_Fit2
                            pop[i, :] = temp
                        case=4.1

                    elif NC_Fit1 < NC_Fit2 and NC_Fit1 < NC_Fit[i]:
                        temp = RP[0, :]
                        if (check_connectivity(RP[0, :], nNode, Rc) == True):
                            pop[i, :] = temp
                            NC_Fit[i] = NC_Fit1
                        case=4.2
                    
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
        if (bestFit == 0):
            break
        if t >= MaxIt:
            break
        print(f"Iteration {t}, case {case}, Best Coverage: {1 - bestFit:.4f}")

    return bestFit, bestSol, Convergence_curve, t
    
