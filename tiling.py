import numpy as np
import itertools


class Tiling(object):
    def __init__(self, n, temperature):
        self.n = n
        self.energy = 0
        self.lattice = self.initialize_lattice()
        self.temperature = temperature
        self.average_energy = 0
        self.average_energy_squared = 0
        self.average_configuration = np.zeros_like(self.lattice, np.double)
        self.correlators = np.zeros_like(self.lattice, np.double)

    def to_3d_lattice(self, lattice):
        n = (lattice[0].size - 1)//2
        result = np.zeros((n, n), dtype=np.int32)
        for i in range(2 * self.n):
            for j in range(2 * self.n):
                if self.is_out_of_border(i, j):
                    continue
                if lattice[i, j] == lattice[i + 1, j + 1]:
                    result[i - lattice[i, j], j - lattice[i, j]] = lattice[i, j]
        return result

    def initialize_lattice(self):
        self.energy = 0
        lattice = np.zeros((2 * self.n + 1, 2 * self.n + 1), np.int64)

        for i in range(self.n+1):
            for j in range(self.n+1):
                lattice[self.n+i, j + i] = i
                lattice[j + i, self.n + i] = i

        for h in range(self.n, 0, -1):
            for i in range(self.n-h, -1, -1):
                for j in range(1, h+1):
                    lattice[self.n-i+j-1][h+i+j-1] = j
                    self.energy += 1
        return lattice

    def is_out_of_border(self, i, j):
        return (i <= 0) or (j <= 0) or (i - j >= self.n) or (j - i >= self.n) or (i >= 2*self.n) or (j >= 2*self.n)

    def is_correct_change(self, i, j, height_difference):
        if self.is_out_of_border(i, j) or (self.lattice[i + 1, j + 1] - self.lattice[i - 1, j - 1] > 1):
            return False
        if ((self.lattice[i, j] + height_difference > self.lattice[i + 1, j]) or (self.lattice[i, j] + height_difference > self.lattice[i, j + 1])
                or (self.lattice[i, j] + height_difference > self.lattice[i + 1, j + 1])):
            return False
        if ((self.lattice[i, j] + height_difference < self.lattice[i - 1, j]) or (self.lattice[i, j] + height_difference < self.lattice[i, j - 1])
                or (self.lattice[i, j] + height_difference < self.lattice[i - 1, j - 1])):
            return False
        return True

    def locally_flippable(self, i, j, height_difference):
        self.lattice[i, j] += height_difference
        params = itertools.product(range(i - 1, i + 2), range(j - 1, j + 2), [-1, 1])
        result = np.count_nonzero([self.is_correct_change(*p) for p in params])
        self.lattice[i, j] -= height_difference
        return result

    def set_temperature(self, temperature):
        self.temperature = temperature

    def change_configuration(self, skip=100):
        height_difference_array = np.where(np.random.rand(skip) > 0.5, 1, -1)
        i_array = np.random.randint(0, 2*self.n + 1, size=skip)
        j_array = np.random.randint(0, 2*self.n + 1, size=skip)
        transition_probabilities = np.random.rand(skip).astype(object)
        states = np.column_stack((i_array, j_array, height_difference_array, transition_probabilities))
        for (i, j, height_difference, p) in states:
            if not self.is_correct_change(i, j, height_difference):
                continue
            if p < np.exp(-height_difference/self.temperature):
                self.lattice[i, j] += height_difference
                self.energy += height_difference

    def metropolis(self, iterations, skip=100, thermalization=100000):
        self.initialize_sample()
        for _ in range(thermalization):
            self.change_configuration(skip)
        print("Thermalization completed")

        for _ in range(iterations):
            self.change_configuration(skip)
            self.sample()
        self.finalize_sample(iterations)

    def initialize_sample(self):
        self.lattice = self.initialize_lattice()
        params = itertools.product(range(0, 2*self.n + 1), range(0, 2*self.n + 1), [-1, 1])
        self.flippable = np.count_nonzero([self.is_correct_change(*p) for p in params])
        self.average_energy = 0
        self.average_energy_squared = 0
        self.average_configuration = np.zeros_like(self.lattice, np.double)
        self.correlators = np.zeros_like(self.lattice, np.double)

    def sample(self):
        self.average_configuration += self.lattice
        self.correlators += self.lattice[self.n, self.n] * self.lattice
        self.average_energy += self.energy
        self.average_energy_squared += self.energy**2

    def finalize_sample(self, iterations):
        self.average_configuration /= iterations
        self.correlators /= iterations
        self.correlators -= self.average_configuration * self.average_configuration[self.n, self.n]
        self.average_energy /= iterations
        self.average_energy_squared /= iterations

    def capacity(self):
        return (self.average_energy_squared - self.average_energy**2)/(self.temperature**2)

