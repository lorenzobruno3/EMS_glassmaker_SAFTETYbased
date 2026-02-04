import numpy as np
import matplotlib.pyplot as plt

# Define the domain
x = np.linspace(-5, 10, 500)

# Define the piecewise function
def f(x):
    return np.piecewise(
        x,
        [x < 0, (x >= 0) & (x <= 2*np.pi), x > 2*np.pi],
        [lambda x: x**2,
         lambda x: np.sin(x),
         lambda x: 2]
    )

y = f(x)
y2 = f(x*1.1)
# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=2)
plt.plot(x*1.1, y2, linewidth=2, linestyle='--')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Piecewise Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()
