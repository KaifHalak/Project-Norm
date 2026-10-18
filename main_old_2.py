
from neuron import Neuron
from synapse import Synapse
from brain import Brain
from concept_lobe import ConceptLobe
from decision_lobe import DecisionLobe


def main():
    # STEP 1: Create the Concept lobe.
    concepts = ConceptLobe()

    concepts.add_concept(
        "C1",
        {"hungry": True, "blue_visible": True}
    )

    concepts.add_concept(
        "C2",
        {"hungry": True, "orange_visible": True}
    )

    # STEP 2: Create our Brain.
    brain = Brain()

    # Register the external concept signals.
    brain.add_input("C1")
    brain.add_input("C2")

    # Create two decision neurons.
    #
    # threshold=0.0 allows any positive activation to fire.
    #
    # relaxation_rate=1.0 resets the stored state to rest
    # after each update. This makes each test encounter
    # independent instead of carrying over past activation.
    brain.add_neuron(
        "Eat",
        Neuron(threshold=0.0, relaxation_rate=1.0)
    )

    brain.add_neuron(
        "Ignore",
        Neuron(threshold=0.0, relaxation_rate=1.0)
    )

    # STEP 3: Connect concepts to decisions.
    #
    # Both concepts have identical initial weights.
    # We are NOT encoding which mushroom is safe.

    for concept_name in ["C1", "C2"]:

        # Excitatory connection to Eat.
        brain.add_synapse(
            concept_name,
            "Eat",
            Synapse(weight=0.3, conn_type=0)
        )

        # Excitatory connection to Ignore.
        brain.add_synapse(
            concept_name,
            "Ignore",
            Synapse(weight=0.4, conn_type=0)
        )

    # STEP 4: Create our Decision lobe.
    decisions = DecisionLobe(brain)

    # STEP 5: Simulate a hungry Norn seeing blue food.
    perception = {
        "hungry": True,
        "blue_visible": True,
        "orange_visible": False
    }

    # Recognize the active concept.
    concept_outputs = concepts.process(perception)

    # Use those concepts to choose an action.
    action, scores = decisions.decide(concept_outputs)

    print("Concept outputs:", concept_outputs)
    print("Decision scores:", scores)
    print("Chosen action:", action)


if __name__ == "__main__":
    main()
