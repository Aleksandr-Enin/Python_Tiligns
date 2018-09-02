import tiling
import pandas as pd
import numpy as np
from multiprocessing import Pool

ITERATIONS = 100000

def write_results(temperatures, energies, capacities, filename):
    df = pd.DataFrame({'temperature': temperatures, 'energy': energies, 'capacity': capacities})
    df.to_csv(filename)


def call_metropolis(tiling):
    tiling.metropolis(ITERATIONS)
    return tiling

if __name__ == '__main__':
    p = Pool(3)
    n = 10
    temperatures = np.linspace(0.5, 10, 50)
    tilings = [tiling.Tiling(n, t) for t in temperatures]
    tilings = p.map(call_metropolis, tilings)
    energies = [tiling.average_energy for tiling in tilings]
    capacities = [tiling.capacity() for tiling in tilings]
    write_results(temperatures, energies, capacities, 'energy_'+str(n)+'.csv')
