
import random

from population import Population
from genome import Genome


class EvolutionSimulation:
    """
    Simulate evolution across multiple generations.

    Every generation:
    1. Norns live and learn.
    2. Survival and fitness are evaluated.
    3. Parents are selected.
    4. Offspring inherit and mutate genes.
    5. A new population is created.
    """

    def __init__(
        self,
        population_size=10,
        ticks_per_generation=20,
        mutation_rate=0.1
    ):

        if population_size < 2:
            raise ValueError(
                "Population must contain at least 2 Norns."
            )

        if ticks_per_generation < 1:
            raise ValueError(
                "Ticks per generation must be positive."
            )

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(
                "Mutation rate must be between 0 and 1."
            )

        self.population_size = population_size
        self.ticks_per_generation = ticks_per_generation
        self.mutation_rate = mutation_rate

        # Create the first population.
        self.population = Population(
            size=population_size
        )

        # Track statistics across generations.
        self.history = []

    def select_parent(self, candidates):
        """
        Select a parent using tournament selection.

        Randomly sample up to 3 eligible Norns,
        then select the one with highest fitness.

        Only candidates supplied to this function
        can be selected.
        """

        if not candidates:
            raise ValueError(
                "Cannot select from an empty candidate list."
            )

        # Select up to 3 random competitors.
        tournament = random.sample(
            candidates,
            k=min(3, len(candidates))
        )

        # Choose the fittest competitor.
        return max(
            tournament,
            key=lambda norn: norn.get_fitness()
        )

    def create_next_generation(self):
        """
        Create the next population through
        inheritance, mutation, and selection.

        Only surviving Norns are allowed to reproduce.
        At least two survivors are required.
        """

        # STEP 1: Find surviving Norns.
        survivors = self.population.get_alive_norns()

        if len(survivors) < 2:
            return None

        # STEP 2: Preserve the genome of the best survivor.
        #
        # This is called elitism.
        # We copy genetic instructions only,
        # NOT learned synaptic weights.
        best_norn = max(
            survivors,
            key=lambda norn: norn.get_fitness()
        )

        next_genomes = [
            Genome(best_norn.genome.genes)
        ]

        # STEP 3: Produce offspring until the
        # new generation reaches the desired size.
        while len(next_genomes) < self.population_size:

            # Select the first parent.
            parent_a = self.select_parent(survivors)

            # Ensure the second parent is a different Norn.
            possible_partners = [
                norn for norn in survivors
                if norn is not parent_a
            ]

            parent_b = self.select_parent(
                possible_partners
            )

            # STEP 4: Genetic crossover.
            child_genome = parent_a.genome.reproduce(
                parent_b.genome
            )

            # STEP 5: Apply random mutation.
            child_genome.mutate(
                mutation_rate=self.mutation_rate
            )

            # Store the child's genome.
            next_genomes.append(child_genome)

        # STEP 6: Construct an entirely new population.
        #
        # The new Norns have fresh brains and bodies,
        # initialized from their inherited genomes.
        return Population(genomes=next_genomes)

    def run(self, generations=5, epsilon=0.2):
        """
        Run evolution for the requested number
        of generations.

        A generation consists of:
        - Several life ticks
        - Fitness evaluation
        - Reproduction (except after the last generation)
        """

        if generations < 1:
            raise ValueError(
                "Generations must be positive."
            )

        # Start with empty statistics for this run.
        self.history = []

        for generation in range(1, generations + 1):

            print(
                f"\n========== GENERATION {generation} =========="
            )

            # STEP 1: Let the population live.
            for _ in range(self.ticks_per_generation):

                # Stop early if everyone has died.
                if not self.population.get_alive_norns():
                    break

                # Advance all surviving Norns.
                self.population.tick(epsilon=epsilon)

            # STEP 2: Gather population statistics.
            norns = self.population.norns
            survivors = self.population.get_alive_norns()

            average_fitness = (
                sum(norn.get_fitness() for norn in norns)
                / len(norns)
            )

            average_learning_rate = (
                sum(
                    norn.genome.get_gene("learning_rate")
                    for norn in norns
                )
                / len(norns)
            )

            average_hunger_rate = (
                sum(
                    norn.genome.get_gene("hunger_rate")
                    for norn in norns
                )
                / len(norns)
            )

            # The best surviving Norn is eligible
            # for reproduction.
            best_survivor_fitness = (
                max(
                    norn.get_fitness()
                    for norn in survivors
                )
                if survivors else None
            )

            # STEP 3: Record this generation.
            statistics = {
                "generation": generation,
                "alive": len(survivors),
                "population_size": len(norns),
                "ticks": self.population.age,
                "average_fitness": average_fitness,
                "best_survivor_fitness": best_survivor_fitness,
                "average_learning_rate": average_learning_rate,
                "average_hunger_rate": average_hunger_rate
            }

            self.history.append(statistics)

            # STEP 4: Display results.
            print(
                f"Survivors: {len(survivors)}/{len(norns)}"
            )

            print(
                f"Ticks completed: {self.population.age}"
            )

            print(
                f"Average fitness: {average_fitness:.3f}"
            )

            print(
                f"Average learning rate: "
                f"{average_learning_rate:.4f}"
            )

            print(
                f"Average hunger rate: "
                f"{average_hunger_rate:.4f}"
            )

            # STEP 5: Stop after the final generation.
            if generation == generations:
                break

            # STEP 6: Produce the next generation.
            new_population = self.create_next_generation()

            # Evolution cannot continue without
            # at least two surviving parents.
            if new_population is None:

                print(
                    "\nNot enough survivors to reproduce."
                )

                print("Evolution simulation ended.")
                break

            # Replace the current generation.
            self.population = new_population

        return self.history
