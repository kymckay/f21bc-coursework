import network
import dataset
import numpy as np

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
        net = network.network.from_list(self.__pos)
        y_pred = net.forward_propagate(dataset.x)

        # Loss is best minimised so negate for fitness
        return -net.get_loss(dataset.y, y_pred)

    def add_informant(self, informant: 'particle') -> None:
        self.__informants.append(informant)

    def update_best(self):
        fitness = self.fitness()

        if fitness > self.__best_fit:
            self.__best = self.__pos
            self.__best_fit = fitness

class swarm:
    def __init__(
        self,
        # Specifies the desired bounds of the search space
        min_values: np.array,
        max_values: np.array,
        swarm_size: int = 10,
        inertia_weight: float = 1,
        cognative_weight: float = 1,
        social_weight: float = 1,
        step_size: float = 0.5,
        num_informants: int = 3,
    ) -> None:
        # Sanity check
        if len(min_values) != len(max_values):
            raise ValueError('PSO dimension bounds are mismatched')

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

    def search(self):
        pass