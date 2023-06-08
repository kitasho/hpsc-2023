import numpy as np
import matplotlib.pyplot as plt
nx = 41
ny = 41
num_files = 11

for i in range(num_files):
    filename = 'result_{}.csv'.format(i * 50)

    data = np.genfromtxt(filename, delimiter=',')

    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    p = data[:, 4]
    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    U = u.reshape((ny, nx))
    V = v.reshape((ny, nx))
    P = p.reshape((ny, nx))
    
    plt.figure()
    plt.contourf(X, Y, P, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2])

    plt.savefig('velocity_field_{}.png'.format(i))

    plt.close()
