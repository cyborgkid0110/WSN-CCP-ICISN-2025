import matplotlib.pyplot as plt
import numpy as np

# Plot the overall best solution after the main loop
fig, (ax, ay) = plt.subplots(1,2)
fig, ax = plt.subplots()

def draw_circle(ax, center, radius, small_radius):
    # Đường bao  phủ của cảm biến
    outline_circle = plt.Circle(center, radius, fill=False, ec='black', lw=0.8, alpha=1)  
    ax.add_artist(outline_circle)
    
    # Phạm vi cảm biến
    large_circle = plt.Circle(center, radius, color='cyan', alpha=0.2)  
    ax.add_artist(large_circle)

    # Hình tròn nhỏ bên trong
    small_circle = plt.Circle(center, small_radius, fill=False, ec='red', lw=1, alpha=0.7)  
    ax.add_artist(small_circle)

small_radius=1 #red circle
def show_sensor_matrix(num_sensor,Rs,overall_best_solution,best_fitness, ban_position):
    for i in range(num_sensor):
        x, y = overall_best_solution[i, :]
        draw_circle(ax,(x,y),Rs,small_radius)
    # Đặt giới hạn cho trục
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_aspect('equal', adjustable='box')

    # Add grid 0 to 100
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, linewidth=0.5)
    ax.set_title(f"Coverage Ratio {round(best_fitness,2)}%")

    # set limit to obstacle
    # ay.set_xlim(0, 100)
    # ay.set_ylim(0, 100)
    # ay.set_aspect('equal', adjustable='box')
    # ay.set_xticks(np.arange(0, 101, 10))
    # ay.set_yticks(np.arange(0, 101, 10))
    # ay.grid(True, linewidth=0.5)
    # ay.invert_yaxis()

    # for bp in ban_position:
    #     ay.plot(bp[0], bp[1], 'o', color='#FF5733')
    # for i in range(num_sensor):
    #     x, y = overall_best_solution[i, :]
    #     draw_circle(ay,(x,y),Rs,small_radius)

    # ay.set_title(f"Coverage Ratio {round(best_fitness,2)}%")
    plt.savefig('graph.pdf', format='pdf', dpi=300)
    plt.show()

