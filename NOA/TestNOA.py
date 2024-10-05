import numpy as np
from NOA import NOA
from Obstacle import create_circle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

num_solutions = 50  # Number of search agents
max_iteration = 15000  # Maximum number of function evaluations
num_sensors = 30
sensing_range = 10
communication_range = 20
min_length = 0
max_length = 100

obstacle_radius = 30
obstacle_map = create_circle(obstacle_radius, max_length)
center_map = max_length // 2

# solution = [x1, y1, x2, y2, x3, y3, ... xn, yn]

# Call the NOA optimization algorithm (Nutcracker Optimizer)
best_score, best_pos, convergence_curve, t = NOA(
    num_solutions, max_iteration,
    num_sensors, sensing_range, communication_range,
    min_length, max_length,
    obstacle_map, obstacle_radius
)

best_pos = best_pos.reshape(-1, 2)

print(f"Best Fitness: {1 - best_score:.4f}")

############################################################################
# Plot the overall best solution after the main loop
############################################################################
fig, ax = plt.subplots()

colors = plt.cm.viridis(np.linspace(0, 1, num_sensors*num_sensors))  # Create colors for the points

obstacle_circle = Circle((center_map, center_map), obstacle_radius, color='black', fill=False, linewidth=1)
ax.add_patch(obstacle_circle)

# Plot points and circles
for i in range(num_sensors):
    x, y = best_pos[i, :]
    # Plot points with color
    ax.plot(x, y, 'ro', markersize=5)
    
    # Add text for the point
    ax.text(x, y, str(i + 1), fontsize=10, ha='right')
    
    # Draw circles with fixed radius
    circle = Circle((x, y), sensing_range, color='k', fill=False, linewidth=1)
    ax.add_patch(circle)

# Draw lines between points within communication range
for i in range(num_sensors):
    for j in range(i + 1, num_sensors):
        x_i, y_i = best_pos[i, :]
        x_j, y_j = best_pos[j, :]
        distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
        if distance <= communication_range:
            ax.plot([x_i, x_j], [y_i, y_j], color=colors[np.random.randint(0,num_sensors*num_sensors-1)], linewidth=1)

# Set limits for x and y axes (100x100)
ax.set_xlim(0, max_length)
ax.set_ylim(0, max_length)

# Ensure aspect ratio is equal for accurate circle display
ax.set_aspect('equal', adjustable='box')

# Enable grid and show plot
plt.grid(True)
plt.show()  # Show the final plot

print(f"Overall Best Fitness: {1 - best_score}")

