import math
import numpy as np
import matplotlib.pyplot as plt

def scaled_symmetric_random_walk(n):
    time_points = np.linspace(0,1, n+1)
    steps = np.random.choice([-1 , 1], size=n)
    path = np.insert(np.cumsum(steps), 0, 0)
    scaled_path = path/np.sqrt(n)

    return time_points, scaled_path

n_values = [10,100,1000]

plt.figure(figsize=(10, 6))
plt.title('Scaled Symmetric Random Walk Paths on [0, 1]')
plt.xlabel('Time (t)')
plt.ylabel('$W^{(n)}(t)$')
plt.grid(True)

for n in n_values:
    time, path = scaled_symmetric_random_walk(n)
    plt.plot(time, path, label=f'n = {n}')

plt.legend()
plt.show

print("The plot has been saved as scaled_random_walk.png")

