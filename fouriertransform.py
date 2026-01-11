import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Define the Shape to Draw ---
# We need a list of complex numbers (x + iy) representing the path.
# Let's generate a parametric "Heart" shape.

def get_heart_path(n_points):
    t = np.linspace(0, 2*np.pi, n_points)
    # Parametric equations for a heart
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    # Combine into complex numbers: z = x + iy
    return x + 1j * y

# Configuration
n_points = 1000       # Resolution of the original shape
n_circles = 50        # How many circles to use (more = more accurate)
                      # Try changing this to 5, 20, or 100 to see the difference!

# Get the path data
path_data = get_heart_path(n_points)

# --- 2. Compute the Discrete Fourier Transform (DFT) ---

# This breaks the path down into frequencies.
# 'coeffs' tells us the radius and starting angle of each circle.
coeffs = np.fft.fft(path_data)
coeffs = coeffs / n_points # Normalize

# We need the frequencies associated with each coefficient
freqs = np.fft.fftfreq(n_points)

# Sort the circles! 
# We generally want to draw the largest circles first (the "fundamental" frequencies)
# so the animation looks stable.
indices = np.argsort(np.abs(coeffs))[::-1] # Sort by magnitude, descending
coeffs = coeffs[indices]
freqs = freqs[indices]

# Limit the number of circles to what we configured
coeffs = coeffs[:n_circles]
freqs = freqs[:n_circles]

# --- 3. Animation Setup ---

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off') # Hide axes

# Set plot limits (adjust slightly larger than the heart)
limit = 25
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit - 5, limit - 5)

# Visual elements
# The "epicycles" (the gray circles)
circle_lines = [ax.plot([], [], 'w-', lw=0.5, alpha=0.3)[0] for _ in range(n_circles)]
# The "radius arms" (the lines connecting centers)
radius_line, = ax.plot([], [], 'w-', lw=1, alpha=0.8)
# The drawing tip (the final path)
draw_path, = ax.plot([], [], 'm-', lw=2) # Magenta line

path_x, path_y = [], []

def update(frame):
    # 'frame' corresponds to time t
    # t goes from 0 to 1 over the course of the animation
    t = frame / n_points 
    
    # Calculate the position of every circle center
    # The formula for a rotating vector is: c * e^(i * 2*pi * f * t)
    
    # We compute terms for ALL circles at this time step
    exponents = np.exp(1j * 2 * np.pi * freqs * t * n_points)
    terms = coeffs * exponents
    
    # Cumulative sum to find the center of each circle
    # [c0, c0+c1, c0+c1+c2, ...]
    centers = np.cumsum(terms)
    # Insert (0,0) at the start so the first circle starts at origin
    centers = np.insert(centers, 0, 0)
    
    # --- Update Visuals ---
    
    # 1. Update the radius arms (the straight line connecting circles)
    radius_line.set_data(np.real(centers), np.imag(centers))
    
    # 2. Update the circles themselves
    # Each circle is centered at centers[i] with radius abs(coeffs[i])
    for i in range(n_circles):
        center = centers[i]
        radius = np.abs(coeffs[i])
        
        # Generate points for a circle
        theta = np.linspace(0, 2*np.pi, 50)
        c_x = np.real(center) + radius * np.cos(theta)
        c_y = np.imag(center) + radius * np.sin(theta)
        
        circle_lines[i].set_data(c_x, c_y)
        
    # 3. Update the drawing path (trail)
    tip = centers[-1]
    path_x.append(np.real(tip))
    path_y.append(np.imag(tip))
    draw_path.set_data(path_x, path_y)
    
    return circle_lines + [radius_line, draw_path]

# Create animation
# frames = n_points corresponds to one full cycle (0 to 2*pi)
anim = FuncAnimation(fig, update, frames=n_points, interval=20, blit=True)

plt.show()