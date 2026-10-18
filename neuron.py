class Neuron:
    
    def __init__(self, state=0.0, threshold=1.0, rest_state=0.0, relaxation_rate=0.2):
        self.state = state  # Current neuron value
        self.threshold = threshold # Min neuron value before emitting signal
        self.rest_state = rest_state # Value the neuron naturally returns to
        self.relaxation_rate = relaxation_rate # How fast the neuron returns to rest


    # U just imagine that 2 inputs from 2 other neurons are going into this one, where 1 is excitation and other is inhibition
    def update(self, excitation, inhibition):

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

def main():
    neuron = Neuron(state=0.4, threshold=0.6, rest_state=0.8, relaxation_rate=0.2)
    
    print(f"Initial neuron state: {neuron.state}")
    
    output = neuron.update(excitation=0.5, inhibition=0.2)
    print(f"Output of neuron: {output}")
    
    new_state = neuron.state
    print(f"Neuron state after update: {new_state}")


if __name__ == "__main__":
    main()