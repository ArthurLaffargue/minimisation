import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')
ax.set_axis_off()

# Make data.
X = np.arange(-2.5, 2.5, 0.05)
Y = np.arange(-2.5, 2.5, 0.05)
X, Y = np.meshgrid(X, Y)
Z =  3*(1-X)**2*np.exp(-(X**2) - (Y+1)**2) \
   - 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2) \
   - 1/3*np.exp(-(X+1)**2 - Y**2)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='jet',antialiased=True,linewidths=0.2,edgecolor="k")


plt.tight_layout()
plt.savefig("3DsurfacePlot.svg",dpi=300)
plt.grid(False)
plt.show()
