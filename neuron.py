class Neuron:
    
    def __init__(self, state=0.0, threshold=1.0, rest_state=0.0, relaxation_rate=0.2):
        """Initialize a neuron from scalar numeric parameters.

        Args:
            state (float): Initial internal activation, e.g. 0.4; default 0.0.
            threshold (float): Output requires activation strictly above this
                value, e.g. 0.6; default 1.0 produces no output after clamping.
            rest_state (float): Target for relaxation, e.g. 0.8; default 0.0.
            relaxation_rate (float): Fraction of the distance to rest applied
                per update, e.g. 0.2; default 0.2. Normally in [0.0, 1.0].

        Returns:
            None. Stores the parameters on this instance without validation.
        """
        self.state = state  # Current neuron value
        self.threshold = threshold # Min neuron value before emitting signal
        self.rest_state = rest_state # Value the neuron naturally returns to
        self.relaxation_rate = relaxation_rate # How fast the neuron returns to rest


    # We need to calculate the sum of excitatory and inhibitionary signals and pass it here. So doesnt matter how many synapses are connected to this neuron
    def update(self, excitation, inhibition):
        """Apply incoming signals, emit an output, then relax the internal state.

        Args:
            excitation (float): Scalar activation added to state, e.g. 0.5.
            inhibition (float): Scalar activation subtracted from state, e.g. 0.2.

        Returns:
            float: Scalar output in [0.0, 1.0], measured before relaxation.
            Returns the clamped activation if it strictly exceeds threshold,
            otherwise 0.0. For state=0.4 and threshold=0.6, inputs (0.5, 0.2)
            produce approximately 0.7. Also updates self.state toward rest_state;
            the relaxed state is not clamped again.
        """

        # 1. Calculate the new state using incoming signals.
        # Clamp the result between 0 and 1.
        new_state = max(
            0.0,
            min(self.state + excitation - inhibition, 1.0)
        )

        # 2. Calculate the neuron's output.
        # Below or at threshold, the output is zero.
        # This does NOT stop the internal state from updating.
        output = (
            new_state if new_state > self.threshold else 0.0
        )
        
        # NOTE: Even if the state does not meet the threshhold requirement, the output will be 0, BUT the internal state will still update. Remember that the output is seperate from internal state.

        # 3. Relax the state toward its rest state.
        # This works whether state is above or below rest. We are "clamping" it: ie. the result will be between 0 and 1
        self.state = new_state + self.relaxation_rate * (
            self.rest_state - new_state
        )

        # 4. Return the signal sent to other neurons.
        return output


