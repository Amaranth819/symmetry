from scipy.stats import vonmises
import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    a = -np.pi
    b = np.pi
    xs = np.linspace(a, b, 100)
    ys = vonmises.pdf(xs, a, b)
    plt.plot(xs, ys)
    plt.savefig('vonmise.png')
    plt.close()