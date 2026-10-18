class Brain:
    
    def __init__(self):
        self.neurons = {} # A: Neuron obk
        self.synapses = [] # [ (source, dest, synapse obj) ]
        
        # Names of input units whose outputs are calculated
        # outside Brain, such as our Concept lobe.
        self.input_names = set()
        
    def add_neuron(self, name, neuron_obj):
        self.neurons[name] = neuron_obj
        
    def add_input(self, name):
        # Register an externally computed input.
        # The Concept lobe will supply its numerical output.
        self.input_names.add(name)

        
    def add_synapse(self, source_name, dest_name, synapse_obj):
        
        if source_name not in self.neurons and source_name not in self.input_names:
            print(f"Error: Source neuron '{source_name}' does not exist.")
            return
        
        if dest_name not in self.neurons:
            print(f"Error: Destination neuron '{dest_name}' does not exist.")
            return
        
        self.synapses.append((source_name, dest_name, synapse_obj))
        

    def propagate(self, source_neurons):
        # source_neurons contains tuples:
        # [("A", outputA), ("B", outputB)]

        # Return an empty dictionary if there are no sources.
        if not source_neurons:
            return {}

        # Store accumulated inputs for target neurons.
        # Example: {"C": 0.35}
        tot_excitation = {}
        tot_inhibition = {}

        # Store target neuron outputs after updating them.
        outputs = {} # "C": 0.0

        # STEP 1: Loop through each source neuron.
        for source_name, neuron_output in source_neurons:

            # STEP 2: Find connections originating from this neuron.
            for conn_source, dest_name, synapse in self.synapses:

                # Skip connections belonging to other neurons.
                if conn_source != source_name:
                    continue

                # STEP 3: Transmit the source output through its synapse.
                excitation, inhibition = synapse.transmit(neuron_output)

                # STEP 4: Accumulate excitation for the destination.
                tot_excitation[dest_name] = (
                    tot_excitation.get(dest_name, 0.0) + excitation
                )

                # Accumulate inhibition for the same destination.
                tot_inhibition[dest_name] = (
                    tot_inhibition.get(dest_name, 0.0) + inhibition
                )

        # STEP 5: Update each target neuron only once.
        for neuron_name in tot_excitation:

            neuron_obj = self.neurons[neuron_name]

            # Combine all incoming excitatory and inhibitory signals.
            neuron_output = neuron_obj.update(
                excitation=tot_excitation[neuron_name],
                inhibition=tot_inhibition.get(neuron_name, 0.0)
            )

            # Store the resulting output using the neuron's name.
            outputs[neuron_name] = neuron_output

        # STEP 6: Return the outputs instead of recursively updating.
        return outputs

    
    def mark_decision_synapses(self, concept_outputs, chosen_action):
        """
        Mark synapses involved in the Norn's selected action.

        concept_outputs:
            {"C1": 1.0, "C2": 0.0}

        chosen_action:
            "Eat" or "Ignore"
            
        Remember that the chosen_action is also a neuron
        """
        
        for source_name, dest_name, synapse_obj in self.synapses:
            if concept_outputs.get(source_name, 0.0) > 0.0 and dest_name == chosen_action:
                synapse_obj.mark_used()
        

    def decay_eligibility(self):
        """
        Allow susceptibility to fade for all connections.
        """

        for synapse in self.synapses:
            synapse[2].decay_susceptibility()


    def apply_reinforcement(self, reward, punishment):
        """
        Update susceptible synaptic weights using
        the provided chemical reinforcement signals.
        """

        for synapse in self.synapses:
            if synapse[2].susceptibility > 0:
                synapse[2].reinforce(reward, punishment)
        
        
    def clear_eligibility(self):
        """
        Clear susceptibility after a completed trial.

        This is appropriate for our independent food trials.
        Later, in a continuous simulation, we'll retain
        susceptibility across ticks for delayed learning.
        """

        # Every synapse has its own susceptibility value.
        for _, _, synapse_obj in self.synapses:
            synapse_obj.susceptibility = 0.0
