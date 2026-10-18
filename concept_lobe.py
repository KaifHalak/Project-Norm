
class ConceptLobe:

    def __init__(self):
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
        
        # Register a new concept with its conditions.
        #
        # name: An identifier such as "C1".
        # conditions: A dictionary of expected sensory values.
        
        if self.concepts.get(name):
            print(f"Warning: Concept '{name}' already exists. Overwriting.")
        
        self.concepts[name] = conditions


    def process(self, perception):
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
                
                 
        
            
            
