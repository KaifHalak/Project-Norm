
import random

from norn import Norn
from environment import Environment


def test_continuous_life():

    # Make random exploration reproducible.
    random.seed(7)

    # STEP 1: Create one Norn.
    #
    # The Norn has its own genome,
    # brain, biochemistry, and receptors.
    norn = Norn()

    # STEP 2: Create its environment.
    environment = Environment()

    print("========== NORN IS BORN ==========")

    print("Age:", norn.age)
    print("Hunger:", norn.body.get_chemical("hunger"))
    print("Pain:", norn.body.get_chemical("pain"))

    # STEP 3: Simulate 40 ticks.
    #
    # We alternate between blue and orange
    # mushrooms so the Norn encounters both.
    for step in range(1, 41):

        # Odd ticks: Blue mushroom.
        # Even ticks: Orange mushroom.
        mushroom_color = (
            "blue" if step % 2 == 1 else "orange"
        )

        # Run one complete life step.
        result = norn.live_step(
            environment=environment,
            mushroom_color=mushroom_color,
            epsilon=0.2
        )

        # Print the first 10 ticks,
        # then every fifth tick.
        if step <= 10 or step % 5 == 0:

            print(
                f"Tick {result['age']:02d} | "
                f"Food={result['mushroom']:6s} | "
                f"Action={result['action']:6s} | "
                f"Hunger={result['hunger']:.2f} | "
                f"Pain={result['pain']:.2f} | "
                f"Reward={result['reward']:.2f} | "
                f"Punishment={result['punishment']:.2f}"
            )

    # STEP 4: Inspect the Norn's final biological state.
    print("\n========== FINAL BODY STATE ==========")

    print("Age:", norn.age)
    print("Hunger:", norn.body.get_chemical("hunger"))
    print("Pain:", norn.body.get_chemical("pain"))

    # STEP 5: Evaluate its neural preferences.
    #
    # We assume the Norn is hungry for both tests
    # so C1 and C2 can activate.
    #
    # This evaluation does not change the actual
    # Norn's biological concentrations or weights.
    print("\n========== FINAL DECISIONS ==========")

    for color in ["blue", "orange"]:

        test_perception = {
            "hungry": True,
            "blue_visible": color == "blue",
            "orange_visible": color == "orange"
        }

        concept_outputs = norn.concepts.process(
            test_perception
        )

        # No random exploration or reinforcement.
        action, scores = norn.decisions.decide(
            concept_outputs
        )

        print(
            f"{color.upper()}: "
            f"Action={action}, "
            f"Scores={scores}"
        )

    # STEP 6: Inspect the learned synaptic weights.
    print("\n========== LEARNED SYNAPSES ==========")

    for source, destination, synapse in norn.brain.synapses:

        connection_type = (
            "Exc" if synapse.conn_type == 0 else "Inh"
        )

        print(
            f"{source} -> {destination} "
            f"({connection_type}): "
            f"STW={synapse.stw:.3f}, "
            f"LTW={synapse.ltw:.3f}"
        )

    # STEP 7: Verify that one Norn lived
    # through all 40 simulation ticks.
    assert norn.age == 40

    print("\nContinuous life simulation completed!")


if __name__ == "__main__":
    test_continuous_life()
