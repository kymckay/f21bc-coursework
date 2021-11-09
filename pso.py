from typing import Iterable
import network
import dataset
import numpy as np
from math import inf

def _get_best_pos(particles: Iterable):
    best_fit = inf

    for p in particles:
        p_best, p_fit = p.get_best()

        # Loss is minimised
        if p_fit < best_fit:
            best_fit = p_fit
            best_pos = p_best

    return best_pos

class particle:
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
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

        # Using loss for fitness since accuracy involved rounding which
        # means improvement will be limited to just getting probabilities
        # on the right side of 0.5
        return net.get_loss(dataset.y, y_pred)

    def add_informant(self, informant: 'particle') -> None:
        self.__informants.append(informant)

    def update_best(self):
        fitness = self.fitness()

        # Loss is minimised
        if fitness < self.__best_fit:
            self.__best = self.__pos
            self.__best_fit = fitness

    # For efficiency, can only update informant best after all particles
    # update their own best (avoids repeatedly finding their fitness)
    def update_informed_best(self):
        self.__inf_best = _get_best_pos(self.__informants)

    def get_best(self) -> np.ndarray:
        return self.__best, self.__best_fit

    def move(self, epsilon, min_bounds, max_bounds):
        new_pos = self.__pos + epsilon * self.__vel

        # Boolean vector of dimensions that are out of bounds
        oob = np.logical_and(min_bounds > new_pos, new_pos < max_bounds)

        # Absorb boundary enforcement, any OOB dimensions get velocity 0
        self.__vel[oob] = 0

        # Absorb boundary enforcement, any OOB coords are bound to limits
        new_pos = np.minimum(np.maximum(min_bounds, new_pos), max_bounds)

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
        min_values: np.ndarray,
        max_values: np.ndarray,
        swarm_size: int = 10,
        num_informants: int = 3,
    ) -> None:
        # Sanity check
        if len(min_values) != len(max_values):
            raise ValueError('PSO dimension bounds are mismatched')

        self.__min_bounds = min_values
        self.__max_bounds = max_values

        # Just some initial values, set via public method
        self.__alpha = 1
        self.__beta = 1
        self.__gamma = 1
        self.__epsilon = 1

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

    def set_hyperparameters(self,
        alpha: float = None, # Inertia weight
        beta: float = None, # Cognative weight
        gamma: float = None, # Social weight
        epsilon: float = None, # Step size
    ) -> None:
        # Only update hyperparameters if asked
        self.__alpha = alpha if alpha is not None else self.__alpha
        self.__beta = beta if beta is not None else self.__beta
        self.__gamma = gamma if gamma is not None else self.__gamma
        self.__epsilon = epsilon if epsilon is not None else self.__epsilon

    def _search_step(self) -> None:
        # Though this may look like additional looping, it is more
        # efficient since updating all the bests first means the fitness
        # values are cached for the informant information sharing step
        for p in self.__swarm:
            p.update_best()

        for p in self.__swarm:
            p.update_informed_best()
            p.steer(self.__alpha, self.__beta, self.__gamma)
            p.move(self.__epsilon, self.__min_bounds, self.__max_bounds)

    def search(self, iterations) -> np.ndarray:
        for _ in range(iterations):
            self._search_step()

        # Return the best fitness particle position
        return _get_best_pos(self.__swarm)

    # Returns positions of all particles throughout the search
    def track_search(self, iterations) -> list[list[np.ndarray]]:
        positions = [[] for _ in self.__swarm]

        for _ in range(iterations):
            self._search_step()

            for i, p in enumerate(self.__swarm):
                positions[i].append(p._particle__pos)

        # Return the particle positions stacked (so index 0 tracks one
        # variable through the iterations)
        return [ np.stack(data, axis=1) for data in positions ]