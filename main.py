import random

from neuron import Neuron
from synapse import Synapse
from brain import Brain
from concept_lobe import ConceptLobe
from decision_lobe import DecisionLobe
from biochemistry import Biochemistry
from environment import Environment
from receptor import Receptor
from emitter import Emitter
from genome import Genome



def environment_feedback(action):
    """
    Temporary test environment.

    Assumption:
    - We are testing a nutritious blue mushroom.
    - Eating it provides positive reinforcement.
    - Ignoring it has no immediate consequence.

    IMPORTANT:
    This knowledge belongs to the ENVIRONMENT,
    not to the Norn's brain.

    Phase 8 will replace this with chemical reactions.
    """

    if action == "Eat":
        return 0.5, 0.0  # reward, punishment

    return 0.0, 0.0



from genome import Genome


def setup_norn(genome=None):
    """
    Create a Norn's neural network using its genome.

    The genome determines:
    - Neuron threshold
    - Relaxation rate
    - Learning rate
    - Initial synaptic weights
    """

    # STEP 1: Create a default genome if none is provided.
    if genome is None:
        genome = Genome()

    # STEP 2: Create the Concept lobe.
    concepts = ConceptLobe()

    # C1 recognizes hungry + blue mushroom.
    concepts.add_concept(
        "C1",
        {"hungry": True, "blue_visible": True}
    )

    # C2 recognizes hungry + orange mushroom.
    concepts.add_concept(
        "C2",
        {"hungry": True, "orange_visible": True}
    )

    # STEP 3: Create the Brain.
    # Its learning rate comes from the genome.
    brain = Brain(
        learning_rate=genome.get_gene("learning_rate")
    )

    # Register Concept lobe outputs as external inputs.
    brain.add_input("C1")
    brain.add_input("C2")

    # STEP 4: Create Decision neurons.
    # Their parameters come from the genome.
    for action in ["Eat", "Ignore"]:

        neuron = Neuron(
            threshold=genome.get_gene("neuron_threshold"),
            relaxation_rate=genome.get_gene("relaxation_rate")
        )

        brain.add_neuron(action, neuron)

    # STEP 5: Create synapses.
    # Each concept connects to Eat and Ignore.
    for concept in ["C1", "C2"]:

        # Excitatory connection to Eat.
        brain.add_synapse(
            concept,
            "Eat",
            Synapse(
                weight=genome.get_gene("eat_exc_weight"),
                conn_type=0
            )
        )

        # Inhibitory connection to Eat.
        brain.add_synapse(
            concept,
            "Eat",
            Synapse(
                weight=genome.get_gene("eat_inh_weight"),
                conn_type=1
            )
        )

        # Excitatory connection to Ignore.
        brain.add_synapse(
            concept,
            "Ignore",
            Synapse(
                weight=genome.get_gene("ignore_weight"),
                conn_type=0
            )
        )

    # STEP 6: Create Decision lobe.
    decisions = DecisionLobe(brain)

    return concepts, decisions, brain



def setup_body(genome):
    """
    Construct a Norn's biochemical system.

    The genome determines how quickly hunger
    increases when its emitter activates.
    """

    # Create the biochemical system.
    body = Biochemistry()

    # Create a hunger emitter using the inherited rate.
    hunger_emitter = Emitter(
        chemical="hunger",
        amount=genome.get_gene("hunger_rate")
    )

    # Create hunger receptor.
    # Threshold remains fixed at 0.5 for now.
    hunger_receptor = Receptor(
        chemical="hunger",
        threshold=0.5
    )

    return body, hunger_emitter, hunger_receptor





def run_trial(concepts, decisions, brain, perception, epsilon=0.2, learning_enabled=True):
    """
    Run one complete learning experience.

    concepts: Our ConceptLobe object.
    decisions: Our DecisionLobe object.
    brain: Our Brain object.
    perception: Current sensory information.
    epsilon: Probability of random exploration.
    """

    # STEP 1:
    # Process perception through the Concept lobe.
    # This identifies which concepts are active.
    
    concept_outputs = concepts.process(perception)

    # STEP 2:
    # Select an action using epsilon-greedy exploration.
    #
    # You should get:
    # action, decision_scores

    action, decision_scores = decisions.select_action(concept_outputs, epsilon)

    # STEP 3:
    # Mark the connections associated with the
    # active concepts and chosen action.
    if learning_enabled:
             brain.mark_decision_synapses(concept_outputs, action)
    


    # STEP 4:
    # Execute the action in our temporary environment.
    #
    # Call environment_feedback(action).
    # It returns reward and punishment.

    reward, punishment = environment_feedback(action)


    # STEP 5:
    # Simulate a delay of two ticks before reinforcement.
    #
    # Call brain.decay_eligibility() twice.
    # For this test, we're isolating eligibility decay;
    # no other simulation systems advance during this delay.

    if learning_enabled:
        brain.decay_eligibility()
        brain.decay_eligibility()
        
        
        # STEP 6:
        # Apply the reward and punishment to the Brain.
    
        brain.apply_reinforcement(reward, punishment)
    
    
        # STEP 7:
        # Loop through every synapse in brain.synapses.
        # Call settle_weights() once for each synapse.
        #
        # Hint:
        # Each element contains:
        # (source_name, destination_name, synapse_obj)
        
        for _, _, synapse_obj in brain.synapses:
            synapse_obj.settle_weights()
    
    
        # STEP 8:
        # Clear eligibility because this independent
        # food encounter has finished.
        # The reaon for this is that the consequence for each action is immediate, so no need to wait
        brain.clear_eligibility()
        
    
    

    # STEP 9:
    # Return the selected action, decision scores,
    # and reward for inspecting the experiment.

    return action, decision_scores, reward



def run_life_trial(
    brain,
    concepts,
    decisions,
    body,
    environment,
    hunger_receptor,
    mushroom_color,
    epsilon=0.2
):
    """
    Simulate one complete food encounter.

    The Norn:
    1. Perceives its environment.
    2. Recognizes concepts.
    3. Chooses an action.
    4. Experiences biological consequences.
    5. Produces reward or punishment.
    6. Updates its synaptic weights.
    """

    # STEP 1: Build the Norn's perception.
    # Hunger comes from its actual chemical concentration.
    # The mushroom color comes from the environment.

    perception = {
        "hungry": hunger_receptor.is_active(body),
        "blue_visible": mushroom_color == "blue",
        "orange_visible": mushroom_color == "orange"
    }

    # STEP 2: Identify active concepts.
    # Example: Hungry + Blue -> C1 = 1.0

    concept_outputs = concepts.process(perception)

    # STEP 3: Choose an action.
    # Epsilon allows the Norn to explore randomly.

    action, decision_scores = decisions.select_action(
        concept_outputs,
        epsilon
    )

    # STEP 4: Mark the synapses responsible for
    # the chosen action as eligible for reinforcement.

    brain.mark_decision_synapses(
        concept_outputs,
        action
    )

    # STEP 5: Save chemical concentrations BEFORE acting.
    # We will compare these with the values after the action.

    old_hunger = body.get_chemical("hunger")
    old_pain = body.get_chemical("pain")

    # STEP 6: Execute the action.
    # Eating blue reduces hunger.
    # Eating orange increases pain.
    # Ignoring does nothing.

    environment.execute_action(
        body,
        action,
        mushroom_color
    )

    # STEP 7: Generate chemical reinforcement.
    # Hunger reduction -> Reward
    # Pain increase -> Punishment

    body.react(old_hunger, old_pain)

    # STEP 8: Read the chemical reinforcement signals.

    reward = body.get_chemical("reward")
    punishment = body.get_chemical("punishment")

    # STEP 9: Simulate two ticks of delay before
    # reinforcement reaches eligible synapses.

    for _ in range(2):
        brain.decay_eligibility()

    # Apply reinforcement using the resulting
    # susceptibility values.

    brain.apply_reinforcement(
        reward,
        punishment
    )

    # STEP 10: Settle short-term and long-term weights.
    # STW moves toward LTW.
    # LTW moves toward STW more slowly.

    for _, _, synapse in brain.synapses:
        synapse.settle_weights()

    # STEP 11: Clear eligibility.
    # This encounter is finished.
    # Learned synaptic weights remain unchanged here.

    brain.clear_eligibility()

    # STEP 12: Return information about the experience.

    return action, reward, punishment



def run_experiment():
    """
    Compare two identical Norns:

    Norn A: Learning enabled.
    Norn B: Learning disabled.

    Goal:
    Determine whether synaptic learning changes the
    Norn's preferred action after repeated experiences.
    """

    # STEP 1: Create two independent Norns.
    # Both start with identical neural configurations.
    concepts_a, decisions_a, brain_a = setup_norn()
    concepts_b, decisions_b, brain_b = setup_norn()

    # Both Norns experience the same situation.
    perception = {
        "hungry": True,
        "blue_visible": True
    }

    # STEP 2: Evaluate both Norns BEFORE training.
    # decide() uses winner-take-all without exploration.
    initial_action_a, initial_scores_a = decisions_a.decide(
        concepts_a.process(perception)
    )

    initial_action_b, initial_scores_b = decisions_b.decide(
        concepts_b.process(perception)
    )

    print("========== BEFORE TRAINING ==========")
    
    # This is when the weights are fresh. To train, we need to run the sim multiple times
    
    print("Norn A:", initial_action_a, initial_scores_a)
    print("Norn B:", initial_action_b, initial_scores_b)

    # STEP 3: Train Norn A with learning enabled.
    # Fix the random seed to reproduce exploration choices.
    random.seed(7)

    for trial in range(30):
        # Each trial selects an action, receives feedback,
        # and updates eligible synaptic weights.
        run_trial(
            concepts_a,
            decisions_a,
            brain_a,
            perception,
            epsilon=0.2,
            learning_enabled=True
        )

    # STEP 4: Train Norn B with learning disabled.
    # Reset the seed so both experiments use the same
    # sequence of random exploration decisions.
    random.seed(7)

    for trial in range(30):
        # This Norn can explore and receive feedback,
        # but its synaptic weights remain unchanged.
        run_trial(
            concepts_b,
            decisions_b,
            brain_b,
            perception,
            epsilon=0.2,
            learning_enabled=False
        )

    # STEP 5: Evaluate both Norns AFTER training.
    # We call decide() directly to disable exploration
    # and avoid any additional learning updates.
    final_action_a, final_scores_a = decisions_a.decide(
        concepts_a.process(perception)
    )

    final_action_b, final_scores_b = decisions_b.decide(
        concepts_b.process(perception)
    )

    print("\n========== AFTER TRAINING ==========")
    print("Norn A:", final_action_a, final_scores_a)
    print("Norn B:", final_action_b, final_scores_b)

    # STEP 6: Inspect learned synaptic weights.
    # We specifically want the excitatory connection
    # from Concept C1 to the Eat decision neuron.

    print("\n========== SYNAPTIC WEIGHTS ==========")

    for label, brain in [("Norn A", brain_a), ("Norn B", brain_b)]:

        # Each stored connection contains:
        # (source_name, destination_name, synapse_object)
        for source, destination, synapse in brain.synapses:

            # Find the C1 -> Eat excitatory connection.
            if (source == "C1"
                    and destination == "Eat"
                    and synapse.conn_type == 0):

                print(
                    f"{label}: "
                    f"STW={synapse.stw:.3f}, "
                    f"LTW={synapse.ltw:.3f}"
                )



def test_mushroom_learning():

    # Make exploration reproducible.
    random.seed(7)

    # STEP 1: Initialize the Norn.
    # We create the brain ONCE so learning is preserved.
    concepts, decisions, brain = setup_norn()

    # Create the environment and hunger receptor.
    environment = Environment()
    hunger_receptor = Receptor("hunger", 0.5)

    # STEP 2: Run 100 food encounters.
    for trial in range(100):

        # Create a fresh body for each encounter.
        #
        # This is a controlled experiment:
        # Hunger starts at 0.8 and pain starts at 0.0.
        #
        # IMPORTANT:
        # We reset the BODY, not the BRAIN.
        # The learned synaptic weights are preserved.
        body = Biochemistry()

        # Alternate between blue and orange mushrooms.
        if trial % 2 == 0:
            color = "blue"
        else:
            color = "orange"

        # Run one complete learning experience.
        action, reward, punishment = run_life_trial(
            brain=brain,
            concepts=concepts,
            decisions=decisions,
            body=body,
            environment=environment,
            hunger_receptor=hunger_receptor,
            mushroom_color=color,
            epsilon=0.2
        )

        # Display some training information.
        if (trial + 1) % 10 == 0:
            print(
                f"Trial {trial + 1}: "
                f"Color={color}, "
                f"Action={action}, "
                f"Reward={reward:.2f}, "
                f"Punishment={punishment:.2f}"
            )

    # STEP 3: Evaluate learning.
    #
    # We disable exploration by calling decide()
    # directly, without using select_action().
    #
    # This shows what the Norn has actually learned.

    print("\n========== FINAL EVALUATION ==========")

    for color in ["blue", "orange"]:

        # Fresh body so hunger is above the threshold.
        body = Biochemistry()

        perception = {
            "hungry": hunger_receptor.is_active(body),
            "blue_visible": color == "blue",
            "orange_visible": color == "orange"
        }

        # Identify active concepts.
        concept_outputs = concepts.process(perception)

        # Choose the preferred action without exploration.
        action, scores = decisions.decide(concept_outputs)

        print(
            f"{color.upper()}: "
            f"Action={action}, Scores={scores}"
        )

    # STEP 4: Inspect the learned synaptic weights.
    print("\n========== LEARNED SYNAPSES ==========")

    for source, destination, synapse in brain.synapses:

        conn_type = (
            "Excitatory" if synapse.conn_type == 0
            else "Inhibitory"
        )

        print(
            f"{source} -> {destination} ({conn_type}): "
            f"STW={synapse.stw:.3f}, "
            f"LTW={synapse.ltw:.3f}"
        )



def test_genome_expression():

    # STEP 1: Create two genomes.
    genome_a = Genome()
    genome_b = Genome()

    # Give Norn B different inherited characteristics.
    genome_b.set_gene("learning_rate", 0.8)
    genome_b.set_gene("eat_exc_weight", 0.6)
    genome_b.set_gene("hunger_rate", 0.1)

    # STEP 2: Construct both neural networks.
    concepts_a, decisions_a, brain_a = setup_norn(genome_a)
    concepts_b, decisions_b, brain_b = setup_norn(genome_b)

    # STEP 3: Construct both biochemical systems.
    body_a, emitter_a, receptor_a = setup_body(genome_a)
    body_b, emitter_b, receptor_b = setup_body(genome_b)

    # STEP 4: Compare inherited learning rates.
    print("========== GENETIC PARAMETERS ==========")

    print("Norn A learning rate:", brain_a.learning_rate)
    print("Norn B learning rate:", brain_b.learning_rate)

    # STEP 5: Test initial neural decisions.
    perception = {
        "hungry": True,
        "blue_visible": True,
        "orange_visible": False
    }

    action_a, scores_a = decisions_a.decide(
        concepts_a.process(perception)
    )

    action_b, scores_b = decisions_b.decide(
        concepts_b.process(perception)
    )

    print("\n========== INITIAL DECISIONS ==========")

    print("Norn A:", action_a, scores_a)
    print("Norn B:", action_b, scores_b)

    # STEP 6: Compare metabolism.
    # Set both Norns' hunger concentrations to 0.4.
    body_a.add_chemical("hunger", -0.4)
    body_b.add_chemical("hunger", -0.4)

    # Activate each Norn's hunger emitter once.
    emitter_a.emit(body_a)
    emitter_b.emit(body_b)

    print("\n========== HUNGER ==========")

    print("Norn A hunger:", body_a.get_chemical("hunger"))
    print("Norn B hunger:", body_b.get_chemical("hunger"))

    # STEP 7: Compare pain decay.
    body_a.add_chemical("pain", 0.5)
    body_b.add_chemical("pain", 0.5)

    body_a.decay_chemical(
        "pain",
        genome_a.get_gene("pain_decay")
    )

    body_b.decay_chemical(
        "pain",
        genome_b.get_gene("pain_decay")
    )

    print("\n========== PAIN ==========")

    print("Norn A pain:", body_a.get_chemical("pain"))
    print("Norn B pain:", body_b.get_chemical("pain"))


# if __name__ == "__main__":
#     run_experiment()


def main():
    # Make random exploration reproducible.
    random.seed(1)

    # STEP 1: Create our existing neural network.
    concepts, decisions, brain = setup_norn()

    # STEP 2: Create the Norn's biological system.
    body = Biochemistry()

    # STEP 3: Create the environment.
    environment = Environment()

    # STEP 4: Create a hunger receptor.
    # It activates when hunger >= 0.5.
    hunger_receptor = Receptor("hunger", 0.5)

    # STEP 5: Display the Norn's initial condition.
    print("========== BEFORE ==========")
    print("Hunger:", body.get_chemical("hunger"))
    print("Pain:", body.get_chemical("pain"))

    # STEP 6: Run one food encounter.
    action, reward, punishment = run_life_trial(
        brain=brain,
        concepts=concepts,
        decisions=decisions,
        body=body,
        environment=environment,
        hunger_receptor=hunger_receptor,
        mushroom_color="blue",
        epsilon=1.0
    )

    # STEP 7: Display the results.
    print("\n========== AFTER ==========")
    print("Chosen action:", action)
    print("Hunger:", body.get_chemical("hunger"))
    print("Pain:", body.get_chemical("pain"))
    print("Reward:", reward)
    print("Punishment:", punishment)

    # STEP 8: Inspect the synaptic weights.
    print("\n========== SYNAPSES ==========")

    for source, destination, synapse in brain.synapses:
        conn_type = (
            "Excitatory" if synapse.conn_type == 0
            else "Inhibitory"
        )

        print(
            f"{source} -> {destination} ({conn_type}): "
            f"STW={synapse.stw:.3f}, "
            f"LTW={synapse.ltw:.3f}"
        )


if __name__ == "__main__":
    test_genome_expression()
