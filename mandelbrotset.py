import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    # 1. Create a grid of complex numbers
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    # 1j is Python's notation for the imaginary unit 'i'
    c = r1 + r2[:,None] * 1j

    # 2. Initialize z and the output image grid
    z = np.zeros_like(c)
    div_time = np.zeros(z.shape, dtype=int)

    # 3. The main loop (Where the magic happens)
    # We iterate z = z^2 + c
    # NumPy does this for all millions of points simultaneously (vectorization)
    mask = np.full(c.shape, True, dtype=bool) # Keep track of points that haven't escaped
    
    for i in range(max_iter):
        z[mask] = z[mask]**2 + c[mask]
        
        # Check which points have "escaped" past a threshold (usually 2.0)
        diverged = np.greater(np.abs(z), 2.0, out=np.full(c.shape, False), where=mask)
        
        # Record the iteration number when they escaped
        div_time[diverged] = i
        
        # Stop calculating for points that have already escaped
        mask[diverged] = False
        
        # Optimization: If everything has escaped, stop early
        if not mask.any(): break

    return div_time

# --- Configuration ---
# A nice zoomed-out view
xmin, xmax, ymin, ymax = -2.0, 0.5, -1.25, 1.25
# Resolution (higher = slower but more detail)
width, height = 2000, 2000
# How deep to check for escapes (higher gives more detailed edges)
max_iter = 256

print("Calculating fractal... (this might take a moment)")
mandel_img = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
print("Calculation done. Rendering...")

# --- Visualization ---
plt.figure(figsize=(12, 12))

# We use imshow to display the grid of escape times as an image
# 'cmap' determines the color scheme. Try: 'magma', 'inferno', 'twilight_shifted', 'bone'
plt.imshow(mandel_img.T, extent=[xmin, xmax, ymin, ymax], cmap='twilight_shifted')

plt.title("The Mandelbrot Set", color='white')
plt.axis('off') # Hide axes for a clean look

# Optional: Save it in high resolution
# plt.savefig("mandelbrot_hd.png", dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()