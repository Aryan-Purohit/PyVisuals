import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Define the Differential Equations ---

def lorenz(xyz, sigma=10, rho=28, beta=8/3):
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def rossler(xyz, a=0.2, b=0.2, c=5.7):
    x, y, z = xyz
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

def chen(xyz, a=40, b=3, c=28):
    x, y, z = xyz
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return np.array([dx, dy, dz])

def aizawa(xyz, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = xyz
    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    dz = c + a * z - (z**3 / 3) - (x**2 + y**2) * (1 + e * z) + f * z * x**3
    return np.array([dx, dy, dz])

# --- 2. Generate Data (Numerical Integration) ---

def generate_data(func, initial_state, steps, dt):
    traj = np.zeros((steps, 3))
    traj[0] = initial_state
    for i in range(steps - 1):
        # Runge-Kutta 4 (RK4) Integration
        # This prevents the "overflow" errors by being much more precise
        current = traj[i]
        
        k1 = func(current)
        k2 = func(current + k1 * dt / 2)
        k3 = func(current + k2 * dt / 2)
        k4 = func(current + k3 * dt)
        
        # Calculate weighted average
        traj[i + 1] = current + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        
    return traj

# Configuration
dt = 0.01
steps = 3000
loop_speed = 10 # Increase to skip frames for faster animation

# Generate the trajectories
data_lorenz = generate_data(lorenz, [0.1, 0, 0], steps, dt)
data_rossler = generate_data(rossler, [0.1, 0, 0], steps, dt)
data_chen = generate_data(chen, [-0.1, 0.5, -0.6], steps, dt)
data_aizawa = generate_data(aizawa, [0.1, 0, 0], steps, dt)

datasets = [data_lorenz, data_rossler, data_chen, data_aizawa]
titles = ["Lorenz", "Rössler", "Chen", "Aizawa"]
colors = ['cyan', 'yellow', 'magenta', 'lime'] # Matching the image colors

# --- 3. Setup the Visualization ---

plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 10))

axes = []
lines = []

# Create 2x2 grid of 3D axes
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    ax.set_title(titles[i], color='white')
    
    # Hide grid and panes for the "clean" look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.axis('off') # Turn off axes completely to match the sleek look
    
    # Initialize empty line
    line, = ax.plot([], [], [], lw=0.8, color=colors[i])
    lines.append(line)
    axes.append(ax)

    # Set initial camera limits based on data range
    data = datasets[i]
    ax.set_xlim(data[:,0].min(), data[:,0].max())
    ax.set_ylim(data[:,1].min(), data[:,1].max())
    ax.set_zlim(data[:,2].min(), data[:,2].max())

# --- 4. Animation Function ---

def update(frame):
    current_step = frame * loop_speed
    
    for i, line in enumerate(lines):
        data = datasets[i]
        # We plot everything up to the current step
        if current_step < len(data):
            line.set_data(data[:current_step, 0], data[:current_step, 1])
            line.set_3d_properties(data[:current_step, 2])
            
            # Optional: Rotate camera slightly for 3D effect
            axes[i].view_init(elev=30, azim=current_step * 0.1)
            
    return lines

# Create animation
anim = FuncAnimation(fig, update, frames=steps//loop_speed, interval=20, blit=False)

plt.tight_layout()
plt.show()