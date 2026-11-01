"""Stateful, dashboard-independent simulation orchestration."""

import random
from copy import deepcopy

from genome import Genome
from population import Population


class SimulationController:
    """Run one population through learning and successive generations."""

    def __init__(
        self,
        population_size=10,
        ticks_per_generation=20,
        mutation_rate=0.1,
        epsilon=0.2
    ):
        if population_size < 2:
            raise ValueError("Population must contain at least 2 Norns.")
        if ticks_per_generation < 1:
            raise ValueError("Ticks per generation must be positive.")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0 and 1.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1.")

        self.population_size = population_size
        self.ticks_per_generation = ticks_per_generation
        self.mutation_rate = mutation_rate
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        """Create a fresh first generation and clear all recorded history."""
        self.population = Population(size=self.population_size)
        self.generation = 1
        self.ticks_in_generation = 0
        self.total_elapsed_ticks = 0
        self.history = []
        self.generation_history = []
        self.last_events = []
        self.finished = False
        self.extinct = False
        return self.snapshot()

    def step(self):
        """Advance the simulation by one tick and return its state snapshot."""
        if self.finished:
            return self.snapshot()

        if not self.population.get_alive_norns():
            self._finish_generation(extinct=True)
            return self.snapshot()

        self.last_events = self.population.tick(epsilon=self.epsilon)
        self.ticks_in_generation += 1
        self.total_elapsed_ticks += 1
        self._record_tick()

        living_count = len(self.population.get_alive_norns())
        if self.ticks_in_generation >= self.ticks_per_generation or living_count < 2:
            self._finish_generation(
                extinct=living_count < 2
            )

        return self.snapshot()

    def advance_ticks(self, count):
        """Advance up to ``count`` ticks, stopping cleanly on extinction."""
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Tick count must be an integer.")
        if count < 0:
            raise ValueError("Tick count cannot be negative.")

        states = []
        for _ in range(count):
            states.append(self.step())
            if self.finished:
                break
        return states

    def select_parent(self, candidates):
        """Select a fit parent using the existing tournament strategy."""
        if not candidates:
            raise ValueError("Cannot select from an empty candidate list.")
        tournament = random.sample(candidates, min(3, len(candidates)))
        return max(tournament, key=lambda norn: norn.get_fitness())

    def _finish_generation(self, extinct=False):
        """Record generation fitness and either reproduce or stop."""
        norns = self.population.norns
        survivors = self.population.get_alive_norns()
        self.generation_history.append({
            "generation": self.generation,
            "alive": len(survivors),
            "population_size": len(norns),
            "ticks": self.total_elapsed_ticks,
            "generation_ticks": self.ticks_in_generation,
            "average_fitness": self._average(
                norn.get_fitness() for norn in norns
            ),
            "best_fitness": max(
                (norn.get_fitness() for norn in norns),
                default=0.0
            ),
            "best_survivor_fitness": max(
                (norn.get_fitness() for norn in survivors),
                default=None
            ),
            "average_learning_rate": self._average(
                norn.genome.get_gene("learning_rate") for norn in norns
            ),
            "average_hunger_rate": self._average(
                norn.genome.get_gene("hunger_rate") for norn in norns
            ),
            "extinct": extinct,
        })

        if extinct or len(survivors) < 2:
            self.extinct = True
            self.finished = True
            return

        next_genomes = [
            Genome(max(survivors, key=lambda norn: norn.get_fitness())
                   .genome.genes)
        ]
        while len(next_genomes) < self.population_size:
            parent_a = self.select_parent(survivors)
            partners = [norn for norn in survivors if norn is not parent_a]
            parent_b = self.select_parent(partners)
            child = parent_a.genome.reproduce(parent_b.genome)
            child.mutate(self.mutation_rate)
            next_genomes.append(child)

        self.population = Population(genomes=next_genomes)
        self.generation += 1
        self.ticks_in_generation = 0
        self.last_events = []

    @staticmethod
    def _average(values):
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    def _record_tick(self):
        norns = self.population.norns
        self.history.append({
            "tick": self.total_elapsed_ticks,
            "generation": self.generation,
            "generation_tick": self.ticks_in_generation,
            "population_size": len(norns),
            "alive": len(self.population.get_alive_norns()),
            "average_health": self._average(norn.health for norn in norns),
            "average_fitness": self._average(
                norn.get_fitness() for norn in norns
            ),
            "events": deepcopy(self.last_events),
            "norns": [self._norn_snapshot(index, norn)
                      for index, norn in enumerate(norns, start=1)],
        })

    @staticmethod
    def _norn_snapshot(index, norn):
        return {
            "norn_id": index,
            "age": norn.age,
            "alive": norn.alive,
            "health": norn.health,
            "fitness": norn.get_fitness(),
            "meals_eaten": norn.meals_eaten,
            "total_reward": norn.total_reward,
            "total_punishment": norn.total_punishment,
            "chemicals": dict(norn.body.chemicals),
            "genes": dict(norn.genome.genes),
            "neurons": {
                name: neuron.state
                for name, neuron in norn.brain.neurons.items()
            },
            "synapses": [
                {
                    "source": source,
                    "destination": destination,
                    "stw": synapse.stw,
                    "ltw": synapse.ltw,
                    "susceptibility": synapse.susceptibility,
                    "type": "inhibitory" if synapse.conn_type else "excitatory",
                }
                for source, destination, synapse in norn.brain.synapses
            ],
        }

    def snapshot(self):
        """Return the current controller state without exposing mutable internals."""
        return {
            "tick": self.total_elapsed_ticks,
            "generation": self.generation,
            "generation_tick": self.ticks_in_generation,
            "population_size": len(self.population.norns),
            "alive": len(self.population.get_alive_norns()),
            "finished": self.finished,
            "extinct": self.extinct,
            "history_length": len(self.history),
        }

    def get_current_norns(self):
        """Return the current population for non-dashboard integrations."""
        return self.population.norns

    @property
    def current_generation(self):
        """Compatibility alias for integrations using an explicit name."""
        return self.generation

    @property
    def total_ticks(self):
        """Compatibility alias for the elapsed simulation tick count."""
        return self.total_elapsed_ticks
