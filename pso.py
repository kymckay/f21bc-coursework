from typing import Iterable
import network
import dataset
import numpy as np

def _get_best_pos(particles: Iterable):
    # Accuracy can't be negative so
    # valid initial value for search
    best_fit = -1

    for p in particles:
        p_best, p_fit = p.get_best()

        # Fitness always maximised
        if p_fit > best_fit:
            best_fit = p_fit
            best_pos = p_best

    return best_pos

class particle:
    def __init__(
        self,
        position: np.array,
        velocity: np.array,
    ) -> None:
        self.__pos = position
        self.__vel = velocity

        # Best position so far
        self.__best = position
        self.__best_fit = self.fitness()

        # Best position known to informants so far
        self.__inf_best = position

        # Informants used to update social knowledge
        self.__informants = []

    def fitness(self):
        net = network.network.from_list(self.__pos, dataset.num_features)
        y_pred = net.forward_propagate(dataset.x)

        # Want to maximise accuracy
        return np.mean(np.around(y_pred) == dataset.y)

    def add_informant(self, informant: 'particle') -> None:
        self.__informants.append(informant)

    def update_best(self):
        fitness = self.fitness()

        if fitness > self.__best_fit:
            self.__best = self.__pos
            self.__best_fit = fitness

    # For efficiency, can only update informant best after all particles
    # update their own best (avoids repeatedly finding their fitness)
    def update_informed_best(self):
        self.__inf_best = _get_best_pos(self.__informants)

    def get_best(self) -> np.array:
        return self.__best, self.__best_fit

    def move(self, epsilon, min_bounds, max_bounds):
        new_pos = self.__pos + epsilon * self.__vel

        # TODO: Other methods of handling bounds
        new_pos = np.minimum(np.maximum(new_pos, min_bounds), max_bounds)

        self.__pos = new_pos

    # Importantly the velocity update apply a random coefficient to each
    # dimensional component for stochastic behaviour
    def steer(self, alpha, beta, gamma):
        dims = len(self.__vel)

        self.__vel = (
            # Inertia component, want to go way already going
            alpha * np.random.rand(dims) * self.__vel +
            # Cognative component, want to explore near best known area
            beta * np.random.rand(dims) * (self.__best - self.__pos) +
            # Social component, want to explore near best group known area
            gamma * np.random.rand(dims) * (self.__inf_best - self.__pos)
        )

class swarm:
    def __init__(
        self,
        # Specifies the desired bounds of the search space
        min_values: np.array,
        max_values: np.array,
        swarm_size: int = 10,
        num_informants: int = 3,
    ) -> None:
        # Sanity check
        if len(min_values) != len(max_values):
            raise ValueError('PSO dimension bounds are mismatched')

        self.__min_bounds = min_values
        self.__max_bounds = max_values

        space_dims = len(min_values)

        # np.random.rand has uniform distribution, distribute one big
        # coordinate list so particles are positioned uniformly to start
        coords = np.random.rand(space_dims * swarm_size)

        # The swarm consists of uniformly distributed particles
        self.__swarm = []
        for i in range(swarm_size):
            # Map initial uniform [0,1) position to dimension bounds
            position = (
                coords[i * space_dims : (i + 1) * space_dims]
                * (max_values - min_values) + min_values
            )

            # Initial velocity is just random
            # TODO: may need to find suitable initial values for this
            velocity = np.random.rand(space_dims)

            self.__swarm.append(particle(
                position,
                velocity,
            ))

        # Every particle has a set of informants that influence search
        for p in self.__swarm:
            # This is a shallow copy so the reference elements persist
            swarm_copy = self.__swarm.copy()

            # Don't want to inform self
            swarm_copy.remove(p)

            # Shuffles the list in place
            np.random.shuffle(swarm_copy)

            for i in range(num_informants):
                p.add_informant(swarm_copy[i])

    def search(
        self,
        alpha: float = 1, # Inertia weight
        beta: float = 1, # Cognative weight
        gamma: float = 1, # Social weight
        epsilon: float = 1, # Step size
    ) -> np.array:
        converged = False
        temp = 0
        while not converged:
            # Though this may look like additional looping, it is more
            # efficient since updating all the bests first means the fitness
            # values are cached for the informant information sharing step
            for p in self.__swarm:
                p.update_best()

            for p in self.__swarm:
                p.update_informed_best()
                p.steer(alpha, beta, gamma)
                p.move(epsilon, self.__min_bounds, self.__max_bounds)

            temp += 1
            converged = temp == 50


        # Return the best fitness particle position
        return _get_best_pos(self.__swarm)