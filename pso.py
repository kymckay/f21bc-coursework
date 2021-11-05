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

        # Best position known to informants so far
        self.__inf_best = position

        # Informants used to update social knowledge
        self.__informants = []

    def add_informant(self, informant: 'particle') -> None:
        self.__informants.append(informant)

class swarm:
    def __init__(
        self,
        # Specifies the desired bounds of the search space
        min_values: np.array,
        max_values: np.array,
        swarm_size:int = 10,
    ) -> None:
        space_dims = len(min_values)

        # np.random.rand has uniform distribution, distribute one big
        # coordinate list so particles are positioned uniformly to start
        coords = np.random.rand(space_dims * swarm_size)

        self.swarm = []
        for i in range(swarm_size):
            # Map initial uniform [0,1) position to dimension bounds
            position = (
                coords[i * space_dims : (i + 1) * space_dims]
                * (max_values - min_values) + min_values
            )

            # Initial velocity is just random
            # TODO: may need to find suitable initial values for this
            velocity = np.random.rand(space_dims)

            self.swarm.append(particle(
                position,
                velocity,
            ))

    def search():
        pass