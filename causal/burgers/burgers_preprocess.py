from burgers_config import *

# analytical solution
def analytical_solution(x, t, Re):
    return (x / (1 + t)) / (1 + np.sqrt((1 + t) / np.exp(Re / 8)) * np.exp(Re * (x**2) / (4 * (1 + t))))

Nx = 256                               # Number of spatial grid points
Mt = 100                               # Number of time snapshots per Reynolds number

# Set up ranges
x = np.linspace(0, 1, Nx)   # x range
t_values = np.linspace(0, 1, Mt)   # t range

from matplotlib.animation import FuncAnimation
# Set up the figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)  # you can adjust this after seeing output
ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_title('Analytical Solution Animation')

# Initialization function
def init():
    line.set_data([], [])
    return line,

# Animation update function
def update(frame):
    t = t_values[frame]
    y = analytical_solution(x, t, Re)
    line.set_data(x, y)
    ax.set_title(f'Analytical Solution at t={t:.2f}')
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=len(t_values),
                    init_func=init, blit=True, interval=100)

# To display in a notebook or interactive window
plt.show()

# # Set up ranges
# x = np.linspace(0, 1, Nx)   # x range
# t = np.linspace(0, 1, Mt)   # t range
# X, T = np.meshgrid(x, t)


# # Evaluate the function on the grid
# Z = analytical_solution(X, T, Re)

# # Create the pcolor plot
# plt.figure(figsize=(6, 5))
# plt.pcolormesh(X, T, Z, shading='auto')
# plt.colorbar(label='Solution value')
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title(f'Analytical solution pcolor plot (Re={Re})')
# plt.show()
sys.exit()