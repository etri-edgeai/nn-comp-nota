import numpy as np
import torch
import matplotlib.pyplot as plt

batch = torch.nn.BatchNorm2d(2)

def plotVectors(vecs, cols, alpha=1):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    plt.figure()
    plt.axvline(x=0, color='#A9A9A9', zorder=0)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)

    for i in range(len(vecs)):
        x = np.concatenate([[0,0],vecs[i]])
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                   alpha=alpha)

def matrixToPlot(matrix, vectorsCol=['#FF9A13', '#1190FF']):
    """
    Modify the unit circle and basis vector by applying a matrix.
    Visualize the effect of the matrix in 2D.

    Parameters
    ----------
    matrix : array-like
        2D matrix to apply to the unit circle.
    vectorsCol : HEX color code
        Color of the basis vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure containing modified unit circle and basis vectors.
    """
    # Unit circle
    x = np.linspace(-1, 1, 100000)
    y = np.sqrt(1-(x**2))

    # Modified unit circle (separate negative and positive parts)
    x1 = matrix[0,0]*x + matrix[0,1]*y
    y1 = matrix[1,0]*x + matrix[1,1]*y
    x1_neg = matrix[0,0]*x - matrix[0,1]*y
    y1_neg = matrix[1,0]*x - matrix[1,1]*y

    # Vectors
    u1 = [matrix[0,0],matrix[1,0]]
    v1 = [matrix[0,1],matrix[1,1]]

    plotVectors([u1, v1], cols=[vectorsCol[0], vectorsCol[1]])

    plt.plot(x1, y1, 'g', alpha=0.5)
    plt.plot(x1_neg, y1_neg, 'g', alpha=0.5)

A = np.array([[3, 7], [5, 2]])
A_norm = (A-A.mean(0))/np.sqrt(A.var(0))

print(f'Unit circle:')
matrixToPlot(np.array([[1, 0], [0, 1]]))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

print(f'Unit circle transformed by A:')
matrixToPlot(A)
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.show()

print(f'Unit circle transformed by B:')
matrixToPlot(A_norm)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()
#
#
U_A, D_A, V_A = np.linalg.svd(A)
print(f"U: {U_A}")
print(f"D: {D_A}")
print(f"V: {V_A}")

U_B, D_B, V_B = np.linalg.svd(A_norm)
print(f"U: {U_B}")
print(f"D: {D_B}")
print(f"V: {V_B}")
#
#
# u1 = [D[0]*U[0,0], D[0]*U[0,1]]
# v1 = [D[1]*U[1,0], D[1]*U[1,1]]
#
# plotVectors([u1, v1], cols=['black', 'black'])
#
# matrixToPlot(A)
#
# plt.text(-5, -4, r"$\sigma_1u_1$", size=18)
# plt.text(-4, 1, r"$\sigma_2u_2$", size=18)
# plt.xlim(-8, 8)
# plt.ylim(-8, 8)
# plt.show()