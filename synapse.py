
class Synapse:

    def __init__(self, weight, conn_type):
        # Validate the connection type.
        # 0 = Excitatory, 1 = Inhibitory.
        if conn_type not in (0, 1):
            raise ValueError("Connection type must be 0 or 1")

        # Short-term weight controls signal transmission.
        self.stw = weight

        # Long-term weight will be used in Phase 6B.
        # Initially, STW and LTW have the same value.
        self.ltw = weight

        # Temporary sensitivity to reward or punishment.
        # Initially zero because this connection has
        # not recently participated in an action.
        self.susceptibility = 0.0

        # Connection type determines how its signal
        # affects the receiving neuron.
        self.conn_type = conn_type

    def mark_used(self):
         self.susceptibility = 1.0

    def decay_susceptibility(self, decay_factor=0.8):
        self.susceptibility *= decay_factor

    def transmit(self, neuron_output):
        """ Transmit the signal from the source neuron to the destination neuron. """
        # Multiply the source output by the current STW.
        # This means reinforcement can change future
        # signal strengths without changing this method.
        signal = neuron_output * self.stw

        if self.conn_type == 0:
            return signal, 0.0
        else:
            return 0.0, signal        

    def reinforce(self, reward, punishment, learning_rate=0.5):
        """ Update the weights based on learning """
        chemical = reward
        if self.conn_type == 1:
            chemical = punishment
        
        change = learning_rate * self.susceptibility * chemical
        self.stw = max(0.0, min(self.stw + change, 1.0))
        
    def settle_weights(self, stw_rate=0.2, ltw_rate=0.05):
        """ Gradually adjust the short-term and long-term weights toward each other. """
        
        new_stw = self.stw + stw_rate * (self.ltw - self.stw)
        new_ltw = self.ltw + ltw_rate * (self.stw - self.ltw)
        
        # We need to do this approach as both calculations rely on the previous values
        self.stw = new_stw
        self.ltw = new_ltw