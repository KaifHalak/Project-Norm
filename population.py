
import random

from genome import Genome
from norn import Norn
from environment import Environment


class Population:
    """
    Manages a population of Norns.

    Responsibilities:
    1. Create multiple Norns.
    2. Give each Norn its own genome.
    3. Advance the population through time.
    4. Track survival.
    5. Report fitness statistics.
    """


    def __init__(self, size=5, genomes=None):
        """
        Create a population.

        size:
            Number of Norns in the initial population.

        genomes:
            Optional list of existing genomes.
            Used when creating a new generation.

        If genomes is None:
            Create random initial genomes.

        Otherwise:
            Construct Norns from the supplied genomes.
        """

        # Every population shares one environment.
        self.environment = Environment()

        # Store all the Norns.
        self.norns = []

        # Current population simulation tick.
        self.age = 0

        # STEP 1: Decide where the genomes come from.
        if genomes is None:

            if size < 1:
                raise ValueError(
                    "Population size must be positive."
                )

            genomes = []

            # Generate the first population.
            for _ in range(size):

                genome = Genome()

                # Create initial genetic variation.
                genome.mutate(mutation_rate=0.5)

                genomes.append(genome)

        else:
            # Use inherited genomes for later generations.
            genomes = list(genomes)

            if not genomes:
                raise ValueError("Genome list cannot be empty.")

        # STEP 2: Construct one Norn per genome.
        #
        # Each Norn creates an independent brain and body.
        for genome in genomes:
            self.norns.append(Norn(genome))

    def get_alive_norns(self):
        """
        Return all Norns that are currently alive.
        """

        return [
            norn
            for norn in self.norns
            if norn.alive
        ]

    def tick(self, epsilon=0.2):
        """
        Advance the entire population by one tick.

        Every living Norn:
        - Encounters the current mushroom type.
        - Makes a decision.
        - Experiences consequences.
        - Learns.
        - Updates health.

        All Norns share the same environment,
        but have independent internal states.
        """

        # Advance population time.
        self.age += 1

        # The environment presents one mushroom color
        # during this tick.
        #
        # For simplicity, each Norn can interact with
        # its own instance of that mushroom type.
        mushroom_color = random.choice([
            "blue",
            "orange"
        ])

        # Store each Norn's activity.
        events = []

        # Advance every living Norn.
        for norn_id, norn in enumerate(self.norns, start=1):

            # Dead Norns no longer perform actions.
            if not norn.alive:
                continue

            # Run one complete life step.
            result = norn.live_step(
                environment=self.environment,
                mushroom_color=mushroom_color,
                epsilon=epsilon
            )

            # Identify which Norn produced this result.
            result["norn_id"] = norn_id

            events.append(result)

        return events

    def run(self, ticks=40, epsilon=0.2):
        """
        Run the population simulation for
        the specified number of ticks.
        """

        for _ in range(ticks):

            # End the simulation if all Norns have died.
            if not self.get_alive_norns():
                print("All Norns have died.")
                break

            # Advance the entire population.
            events = self.tick(epsilon=epsilon)

            # Print progress every 10 ticks.
            if self.age % 10 == 0:

                alive_count = len(self.get_alive_norns())

                print(
                    f"Tick {self.age}: "
                    f"Alive={alive_count}/{len(self.norns)}"
                )

                # Show each living Norn's current condition.
                for norn_id, norn in enumerate(
                    self.norns, start=1
                ):

                    if not norn.alive:
                        continue

                    print(
                        f"  Norn {norn_id}: "
                        f"Health={norn.health:.3f}, "
                        f"Hunger={norn.body.get_chemical('hunger'):.3f}, "
                        f"Fitness={norn.get_fitness():.2f}"
                    )

    def print_summary(self):
        """
        Display final survival and fitness statistics.
        """

        print("\n========== POPULATION SUMMARY ==========")

        alive_count = len(self.get_alive_norns())

        print("Total Norns:", len(self.norns))
        print("Alive:", alive_count)
        print("Dead:", len(self.norns) - alive_count)

        # Display each Norn's statistics.
        for norn_id, norn in enumerate(self.norns, start=1):

            status = "Alive" if norn.alive else "Dead"

            print(
                f"\nNorn {norn_id}:"
                f"\n  Status: {status}"
                f"\n  Age: {norn.age}"
                f"\n  Health: {norn.health:.3f}"
                f"\n  Meals eaten: {norn.meals_eaten}"
                f"\n  Total reward: {norn.total_reward:.3f}"
                f"\n  Total punishment: {norn.total_punishment:.3f}"
                f"\n  Fitness: {norn.get_fitness():.2f}"
            )

        # Rank Norns by fitness.
        ranked_norns = sorted(
            enumerate(self.norns, start=1),
            key=lambda item: item[1].get_fitness(),
            reverse=True
        )

        print("\n========== FITNESS RANKING ==========")

        for rank, (norn_id, norn) in enumerate(
            ranked_norns, start=1
        ):

            print(
                f"{rank}. Norn {norn_id} "
                f"- Fitness={norn.get_fitness():.2f}"
            )
