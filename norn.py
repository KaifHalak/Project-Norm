
from genome import Genome
from neuron import Neuron
from synapse import Synapse
from brain import Brain
from concept_lobe import ConceptLobe
from decision_lobe import DecisionLobe
from biochemistry import Biochemistry
from emitter import Emitter
from receptor import Receptor


class Norn:
    """
    Represents one living Norn.

    Each Norn owns:
    - A genome
    - A brain with learned synaptic weights
    - A biochemical system
    - Emitters and receptors
    - An age
    """

    def __init__(self, genome=None):

        # STEP 1: Assign a genome.
        #
        # If none is provided, create a default genome.
        # If one is provided, copy its genetic instructions.
        # This prevents two Norns from sharing a mutable genome.

        if genome is None:
            self.genome = Genome()
        else:
            self.genome = Genome(genome.genes)

        # STEP 2: Initialize the Norn's age.
        self.age = 0

        # STEP 3: Construct its biochemical system.
        self.body = Biochemistry()

        # The inherited hunger rate determines how
        # much hunger the emitter releases.
        self.hunger_emitter = Emitter(
            chemical="hunger",
            amount=self.genome.get_gene("hunger_rate")
        )

        # Receptor detects when hunger reaches 0.5.
        self.hunger_receptor = Receptor(
            chemical="hunger",
            threshold=0.5
        )

        # STEP 4: Construct the Concept lobe.
        self.concepts = ConceptLobe()

        # C1: Hungry and blue mushroom visible.
        self.concepts.add_concept(
            "C1",
            {"hungry": True, "blue_visible": True}
        )

        # C2: Hungry and orange mushroom visible.
        self.concepts.add_concept(
            "C2",
            {"hungry": True, "orange_visible": True}
        )

        # STEP 5: Construct the Brain using inherited
        # learning parameters.
        self.brain = Brain(
            learning_rate=self.genome.get_gene("learning_rate")
        )

        # Register external concept inputs.
        self.brain.add_input("C1")
        self.brain.add_input("C2")

        # STEP 6: Create the Decision neurons.
        for action in ["Eat", "Ignore"]:

            self.brain.add_neuron(
                action,
                Neuron(
                    threshold=self.genome.get_gene(
                        "neuron_threshold"
                    ),
                    relaxation_rate=self.genome.get_gene(
                        "relaxation_rate"
                    )
                )
            )

        # STEP 7: Construct synapses using the genome.
        for concept in ["C1", "C2"]:

            # Excitatory connection to Eat.
            self.brain.add_synapse(
                concept,
                "Eat",
                Synapse(
                    weight=self.genome.get_gene(
                        "eat_exc_weight"
                    ),
                    conn_type=0
                )
            )

            # Inhibitory connection to Eat.
            self.brain.add_synapse(
                concept,
                "Eat",
                Synapse(
                    weight=self.genome.get_gene(
                        "eat_inh_weight"
                    ),
                    conn_type=1
                )
            )

            # Excitatory connection to Ignore.
            self.brain.add_synapse(
                concept,
                "Ignore",
                Synapse(
                    weight=self.genome.get_gene(
                        "ignore_weight"
                    ),
                    conn_type=0
                )
            )

        # STEP 8: Create the Decision lobe.
        self.decisions = DecisionLobe(self.brain)

    def perceive(self, mushroom_color):
        """
        Construct a perception dictionary.

        Internal hunger comes from the receptor.
        External mushroom color comes from the environment.
        """

        return {
            "hungry": self.hunger_receptor.is_active(
                self.body
            ),
            "blue_visible": mushroom_color == "blue",
            "orange_visible": mushroom_color == "orange"
        }

    def decide(self, mushroom_color):
        """
        Make a decision without random exploration.

        This is useful for checking the Norn's
        current neural preferences.
        """

        # Read internal and external conditions.
        perception = self.perceive(mushroom_color)

        # Activate matching concepts.
        concept_outputs = self.concepts.process(
            perception
        )

        # Calculate Decision neuron outputs
        # and select the strongest action.
        return self.decisions.decide(concept_outputs)
    
    
    def tick(self):
        """
        Advance the Norn's biological and neural systems
        by one simulation tick.

        Processes:
        1. Increase age.
        2. Increase hunger through the emitter.
        3. Gradually reduce pain.
        4. Advance the brain's time-dependent processes.
        """

        # STEP 1: The Norn becomes one tick older.
        self.age += 1

        # STEP 2: The Norn becomes hungrier over time.
        #
        # The amount released depends on its genome.
        #
        # Example:
        # hunger_rate = 0.05
        # Hunger 0.40 -> 0.45
        self.hunger_emitter.emit(self.body)

        # STEP 3: Pain gradually decreases.
        #
        # The decay factor comes from the genome.
        #
        # Example:
        # pain_decay = 0.9
        # Pain 0.50 -> 0.45
        self.body.decay_chemical(
            "pain",
            factor=self.genome.get_gene("pain_decay")
        )

        # STEP 4: Update the brain.
        #
        # This:
        # - Decays synaptic susceptibility.
        # - Settles STW and LTW.
        self.brain.tick()
        
    
    def live_step(self, environment, mushroom_color, epsilon=0.2):
        """
        Simulate one complete step of the Norn's life.

        Each step:
        1. Advances time.
        2. Perceives the environment.
        3. Selects an action.
        4. Executes the action.
        5. Produces biological consequences.
        6. Applies synaptic learning.

        The brain and body persist across steps.
        """

        # STEP 1: Advance simulation time.
        #
        # Our existing tick() method:
        # - Increases age.
        # - Increases hunger.
        # - Decreases pain.
        # - Decays susceptibility.
        # - Settles STW and LTW.
        self.tick()

        # STEP 2: Perceive the environment.
        #
        # Internal hunger is detected by the receptor.
        # Mushroom color comes from the environment.
        perception = self.perceive(mushroom_color)

        # STEP 3: Activate matching concepts.
        #
        # C1 = Hungry + Blue
        # C2 = Hungry + Orange
        concept_outputs = self.concepts.process(perception)

        # STEP 4: Choose an action.
        #
        # If no concept activates, our simplified
        # brain has no recognized situation.
        # For now, the Norn defaults to Ignore.

        if not any(value > 0 for value in concept_outputs.values()):

            action = "Ignore"

            decision_scores = {
                "Eat": 0.0,
                "Ignore": 0.0
            }

        else:
            # Select an action with epsilon-greedy exploration.
            action, decision_scores = self.decisions.select_action(
                concept_outputs,
                epsilon=epsilon
            )

            # If no decision neuron fires, do nothing.
            if action is None:
                action = "Ignore"

            # Mark connections associated with the
            # active concepts and selected action.
            self.brain.mark_decision_synapses(
                concept_outputs,
                action
            )

        # STEP 5: Save biological state BEFORE the action.
        old_hunger = self.body.get_chemical("hunger")
        old_pain = self.body.get_chemical("pain")

        # STEP 6: Execute the action.
        #
        # Blue mushroom:
        #   Eating reduces hunger.
        #
        # Orange mushroom:
        #   Eating increases pain.
        #
        # Ignore:
        #   No immediate biological effect.
        environment.execute_action(
            self.body,
            action,
            mushroom_color
        )

        # STEP 7: Generate biochemical reinforcement.
        #
        # Hunger reduction -> Reward
        # Pain increase    -> Punishment
        self.body.react(old_hunger, old_pain)

        # STEP 8: Read reinforcement chemical concentrations.
        reward = self.body.get_chemical("reward")
        punishment = self.body.get_chemical("punishment")

        # STEP 9: Apply reinforcement immediately.
        #
        # The action's consequences are immediate
        # in our current environment.
        #
        # Therefore, we don't simulate an additional
        # two-tick delay as in our earlier experiments.
        self.brain.apply_reinforcement(
            reward,
            punishment
        )

        # STEP 10: Clear eligibility after the encounter.
        #
        # This prevents the next mushroom's reinforcement
        # from accidentally affecting the previous action.
        #
        # STW and LTW are NOT erased.
        self.brain.clear_eligibility()

        # STEP 11: Return information about this life step.
        return {
            "age": self.age,
            "mushroom": mushroom_color,
            "action": action,
            "hunger": self.body.get_chemical("hunger"),
            "pain": self.body.get_chemical("pain"),
            "reward": reward,
            "punishment": punishment,
            "decision_scores": decision_scores
        }


