import random

from neuron import Neuron
from synapse import Synapse
from brain import Brain
from concept_lobe import ConceptLobe
from decision_lobe import DecisionLobe



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



def setup_norn():
    """
    Create our first simple learning Norn.

    The initial connections favor Ignore.
    Learning should eventually increase the
    preference for Eat.
    """

    concepts = ConceptLobe()

    # Register the recognizable situation.
    concepts.add_concept(
        "C1",
        {"hungry": True, "blue_visible": True}
    )

    brain = Brain()

    # C1 is calculated externally by ConceptLobe.
    brain.add_input("C1")

    # Decision neurons represent available actions.
    #
    # threshold=0.0 means any positive state can fire.
    # relaxation_rate=1.0 resets the stored state
    # after each update, keeping trials independent.
    brain.add_neuron(
        "Eat",
        Neuron(threshold=0.0, relaxation_rate=1.0)
    )

    brain.add_neuron(
        "Ignore",
        Neuron(threshold=0.0, relaxation_rate=1.0)
    )

    # The Eat decision receives both excitation
    # and inhibition from Concept C1.
    brain.add_synapse(
        "C1", "Eat",
        Synapse(weight=0.3, conn_type=0)
    )

    brain.add_synapse(
        "C1", "Eat",
        Synapse(weight=0.1, conn_type=1)
    )

    # Ignore initially has greater activation.
    brain.add_synapse(
        "C1", "Ignore",
        Synapse(weight=0.4, conn_type=0)
    )

    decisions = DecisionLobe(brain)

    return concepts, decisions, brain


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
    
        brain.clear_eligibility()
        
    
    

    # STEP 9:
    # Return the selected action, decision scores,
    # and reward for inspecting the experiment.

    return action, decision_scores, reward




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


if __name__ == "__main__":
    run_experiment()


def main():
    # Use a fixed seed for reproducible exploration.
    random.seed(7)

    # Create our learning Norn only once.
    concepts, decisions, brain = setup_norn()

    # The situation remains the same for this experiment.
    perception = {
        "hungry": True,
        "blue_visible": True
    }

    # Check the initial preference without exploration.
    initial_action, initial_scores = decisions.decide(
        concepts.process(perception)
    )

    print("Before learning:", initial_scores)
    print("Initially preferred:", initial_action)
    print()

    # Repeat the experience 30 times.
    for trial in range(1, 31):

        # Each trial may select Eat or Ignore.
        # The learning updates happen inside run_trial().
        action, scores, reward = run_trial(
            concepts,
            decisions,
            brain,
            perception,
            epsilon=0.2
        )

        # Print every fifth trial for easier inspection.
        if trial % 5 == 0:
            print(
                f"Trial {trial}: "
                f"Action={action}, Reward={reward:.2f}"
            )

    # Test the learned preference with exploration disabled.
    final_action, final_scores = decisions.decide(
        concepts.process(perception)
    )

    print()
    print("After learning:", final_scores)
    print("Finally preferred:", final_action)


if __name__ == "__main__":
    main()
