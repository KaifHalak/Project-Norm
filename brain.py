class Brain:
    
    def __init__(self, learning_rate=0.5):
        """Create an empty brain.

        Args:
            No explicit parameters.

        Returns:
            None. Initializes neuron and connection collections on this instance.
        """
        self.neurons = {} # A: Neuron obk
        self.synapses = [] # [ (source, dest, synapse obj) ]
        
        # Names of input units whose outputs are calculated
        # outside Brain, such as our Concept lobe.
        self.input_names = set()
        
        self.learning_rate = learning_rate
        
    def add_neuron(self, name, neuron_obj):
        """Register a neuron, replacing any existing entry with the same name.

        Args:
            name (str): Neuron identifier, for example "Eat".
            neuron_obj (Neuron): Instance whose update(excitation, inhibition)
                method returns a scalar activation.

        Returns:
            None. Stores neuron_obj in self.neurons[name].
        """
        self.neurons[name] = neuron_obj
        
    def add_input(self, name):
        """Register the name of an externally computed signal.

        Args:
            name (str): Input identifier, for example "C1".

        Returns:
            None. Adds name to self.input_names.
        """
        # Register an externally computed input.
        # The Concept lobe will supply its numerical output.
        self.input_names.add(name)

        
    def add_synapse(self, source_name, dest_name, synapse_obj):
        """Connect a registered source to a registered destination neuron.

        Args:
            source_name (str): Registered neuron or input name, e.g. "C1".
            dest_name (str): Registered neuron name, e.g. "Eat".
            synapse_obj (Synapse): Connection providing transmit(output) and
                the eligibility and reinforcement methods used by this brain.

        Returns:
            None. Appends (source_name, dest_name, synapse_obj) to self.synapses.
            If either name is invalid, prints an error and adds no connection.
        """
        
        if source_name not in self.neurons and source_name not in self.input_names:
            print(f"Error: Source neuron '{source_name}' does not exist.")
            return
        
        if dest_name not in self.neurons:
            print(f"Error: Destination neuron '{dest_name}' does not exist.")
            return
        
        self.synapses.append((source_name, dest_name, synapse_obj))
        

    def propagate(self, source_neurons):
        """Propagate one batch of signals and update each reached neuron once.

        Args:
            source_neurons (iterable[tuple[str, float]]): Source names paired
                with scalar activations, e.g. [("C1", 1.0), ("C2", 0.0)].

        Returns:
            dict[str, float]: Reached destination names mapped to scalar
            outputs, e.g. {"Eat": 0.7, "Ignore": 0.0}. Returns {} when no
            connections are reached. Updates destination neurons' internal states.
        """
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
        """Mark connections from active concepts to the selected action.

        Args:
            concept_outputs (dict[str, float]): Concept activations, e.g.
                {"C1": 1.0, "C2": 0.0}; only values greater than zero qualify.
            chosen_action (str | None): Destination neuron name, e.g. "Eat"
                or "Ignore". None matches no normally named action neuron.

        Returns:
            None. Sets susceptibility to 1.0 on matching connections.
        """
        
        for source_name, dest_name, synapse_obj in self.synapses:
            if concept_outputs.get(source_name, 0.0) > 0.0 and dest_name == chosen_action:
                synapse_obj.mark_used()
        

    def decay_eligibility(self):
        """Allow susceptibility to fade for all connections.

        Args:
            No explicit parameters; uses self.synapses.

        Returns:
            None. Decays each connection's susceptibility in place using its
            default decay factor (0.8 for Synapse).
        """

        for synapse in self.synapses:
            synapse[2].decay_susceptibility()


    def apply_reinforcement(self, reward, punishment):
        """Update susceptible connections using chemical reinforcement signals.

        Args:
            reward (float): Scalar reward signal, e.g. 0.4.
            punishment (float): Scalar punishment signal, e.g. 0.3.

        Returns:
            None. Reinforces connections with positive susceptibility in place;
            excitatory connections use reward and inhibitory ones use punishment.
        """

        for synapse in self.synapses:
            if synapse[2].susceptibility > 0:
                synapse[2].reinforce(reward, punishment,  learning_rate=self.learning_rate)
        
        
    def clear_eligibility(self):
        """Clear susceptibility after a completed trial.

        Args:
            No explicit parameters; uses self.synapses.

        Returns:
            None. Sets every connection's susceptibility to 0.0.

        This is appropriate for our independent food trials.
        Later, in a continuous simulation, we'll retain
        susceptibility across ticks for delayed learning.
        """

        # Every synapse has its own susceptibility value.
        for _, _, synapse_obj in self.synapses:
            synapse_obj.susceptibility = 0.0
            
    
    def tick(self):
        """
        Advance the brain's time-dependent processes
        by one simulation tick.

        Every tick:
        1. Synaptic susceptibility gradually decreases.
        2. STW and LTW gradually move toward each other.

        Neither process requires new reward or punishment.
        """

        # STEP 1: Decay susceptibility for every synapse.
        # This reduces the influence of older actions.
        self.decay_eligibility()

        # STEP 2: Update short-term and long-term weights.
        for _, _, synapse in self.synapses:
            synapse.settle_weights()

