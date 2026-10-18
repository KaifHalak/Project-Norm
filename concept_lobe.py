
class ConceptLobe:

    def __init__(self):
        """Create an empty concept registry.

        Args:
            No explicit parameters.

        Returns:
            None. Initializes self.concepts as an empty dictionary.
        """
        # Store concept names and the conditions
        # required for each concept to activate.
        #
        # Example:
        # {
        #     "C1": {"hungry": True, "blue_visible": True}
        # }
        # These are AND conditions
        self.concepts = {}

    def add_concept(self, name, conditions):
        """Register the conditions that must all match for a concept to activate.

        Args:
            name (str): Concept identifier, e.g. "C1".
            conditions (dict[str, object]): Expected sensory values, e.g.
                {"hungry": True, "blue_visible": True}. Stored by reference.

        Returns:
            None. Stores or replaces self.concepts[name]. Prints a warning
            when the previous conditions dictionary is nonempty.
        """
        
        # Register a new concept with its conditions.
        #
        # name: An identifier such as "C1".
        # conditions: A dictionary of expected sensory values.
        
        if self.concepts.get(name):
            print(f"Warning: Concept '{name}' already exists. Overwriting.")
        
        self.concepts[name] = conditions


    def process(self, perception):
        """Match perception values against every registered concept.

        Args:
            perception (dict[str, object]): Observed sensory values, e.g.
                {"hungry": True, "blue_visible": False}. Missing keys are
                compared as None; additional keys are ignored.

        Returns:
            dict[str, float]: One entry per concept, e.g. {"C1": 0.0, "C2": 1.0}.
            Values are 1.0 when all conditions match and 0.0 otherwise. A concept
            with no conditions is active; an empty registry returns {}.
        """
        all_concept_names = self.concepts.keys()
        
        final_out = {}
        for concept_name in all_concept_names:
            final_out[concept_name] = 0.0
        
        for concept_name, concept_conditions in self.concepts.items():
            
            all_conditions = len(concept_conditions)
            conditions_met = 0
            
            for condition_name, condition_val in concept_conditions.items():
                if perception.get(condition_name) != condition_val:
                    continue
            
                conditions_met += 1
            
            # We want to check against all conditions, not just the first one that matches
            if conditions_met == all_conditions:
                final_out[concept_name] = 1.0
                
        
        return final_out
                
                 
        
            
            
