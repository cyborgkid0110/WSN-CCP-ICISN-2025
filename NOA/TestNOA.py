from NOA import NOA
import numpy as np
from DrawSensor import show_sensor_matrix
from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
import csv

nPop = 25  # Number of search agents
MaxIt = 500  # Maximum number of function evaluations
nNode = 60
Rs = 10
Rc = 15
VarMin = 0
VarMax = 100

# Call the NOA optimization algorithm (Nutcracker Optimizer)
best_score, bestSol, convergence_curve, t = NOA(
    nPop, MaxIt,
    nNode, Rs, Rc,
    VarMin, VarMax)

bestSol = np.array(bestSol).reshape(-1, 2)

print(f"Best Fitness: {1 - best_score:.4f}")

with open('FOA.CSV', mode ='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([1 - best_score])
    csv_writer.writerows(bestSol)

show_sensor_matrix(nNode, Rs, bestSol, (1 -best_score) * 100)

