from brain import Brain
import random

class DecisionLobe:

    def __init__(self, brain):
        """Attach decision selection to an existing brain.

        Args:
            brain (Brain): Instance containing concept-to-action connections
                and action neurons such as "Eat" and "Ignore".

        Returns:
            None. Stores brain by reference and initializes the action names.
        """
        # Reuse the existing Brain object, which manages
        # neurons, synapses, and signal propagation.
        self.brain = brain

        # Names of the Decision neurons.
        self.actions = ["Eat", "Ignore"]

    def decide(self, concept_outputs):
        """Select the action with the highest strictly positive activation.

        Args:
            concept_outputs (dict[str, float]): Concept activations, e.g.
                {"C1": 1.0, "C2": 0.0}.

        Returns:
            tuple[str | None, dict[str, float]]: (winning_action, scores), e.g.
            ("Eat", {"Eat": 0.7, "Ignore": 0.0}). The winner is None if no score
            exceeds zero. Ties favor the first action in self.actions. Propagating
            the signals updates the connected neurons' internal states.
        """

        # Convert concept outputs into (name, output) tuples.
        # Example:
        # {"C1": 1.0} -> [("C1", 1.0)]
        formatted_outputs = list(concept_outputs.items())

        # Propagate concept signals through the Brain.
        #
        # Keep the dictionary so each output remains
        # associated with its actual neuron name.
        outputs = self.brain.propagate(formatted_outputs)

        # Track the action with the highest activation.
        # Starting at zero prevents inactive actions
        # from winning.
        max_value = 0.0
        max_action = None

        decision_scores = {}

        # Check each available action.
        for action in self.actions:

            # Retrieve this action's output.
            # Default to zero if Brain returned no output.
            score = outputs.get(action, 0.0)

            # Store the score for debugging and inspection.
            decision_scores[action] = score

            # Winner-take-all:
            # Select the action with the largest output.
            if score > max_value:
                max_value = score
                max_action = action

        # Return the winning action and all scores.
        return max_action, decision_scores  


    def select_action(self, concept_outputs, epsilon=0.2):
        """Choose an action using epsilon-greedy exploration.

        Args:
            concept_outputs (dict[str, float]): Concept activations, e.g.
                {"C1": 1.0, "C2": 0.0}.
            epsilon (float): Exploration probability, normally in [0.0, 1.0];
                defaults to 0.2. The range is not validated. Exploration occurs
                when random.random() <= epsilon.

        Returns:
            tuple[str | None, dict[str, float]]: (chosen_action, decision_scores),
            e.g. ("Ignore", {"Eat": 0.7, "Ignore": 0.0}). Scores come from decide
            even when exploration selects a different action. Without exploration,
            the action may be None if no score exceeds zero. Updates brain state.
        """
        
        max_action, decision_scores = self.decide(concept_outputs)
        rand_val = random.random()
        
        if rand_val > epsilon:
            return max_action, decision_scores
        else:
            # Randomly select an action from the available actions.
            random_action = random.choice(self.actions)
            return random_action, decision_scores
            
        
        


        
