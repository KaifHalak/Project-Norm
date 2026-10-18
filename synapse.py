
class Synapse:

    def __init__(self, weight, conn_type):
        """Initialize a weighted excitatory or inhibitory connection.

        Args:
            weight (float): Initial short-term and long-term weight, e.g. 0.5.
                Normally in [0.0, 1.0]; this range is not validated.
            conn_type (int): 0 for excitation or 1 for inhibition. Other values
                raise ValueError.

        Returns:
            None. Initializes stw and ltw to weight and susceptibility to 0.0.
        """
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
         """Mark this connection as eligible for reinforcement.

         Args:
             No explicit parameters.

         Returns:
             None. Sets self.susceptibility to 1.0.
         """
         self.susceptibility = 1.0

    def decay_susceptibility(self, decay_factor=0.8):
        """Scale the connection's current eligibility.

        Args:
            decay_factor (float): Scalar multiplier, default 0.8. Values in
                [0.0, 1.0] fade eligibility; the range is not validated.

        Returns:
            None. Multiplies self.susceptibility by decay_factor in place.
        """
        self.susceptibility *= decay_factor

    def transmit(self, neuron_output):
        """Convert a source activation into excitation and inhibition.

        Args:
            neuron_output (float): Scalar source activation, e.g. 0.8.

        Returns:
            tuple[float, float]: (excitation, inhibition). With stw=0.5 and
            neuron_output=0.8, returns (0.4, 0.0) for conn_type=0 or (0.0, 0.4)
            for conn_type=1. The weighted signal is not clamped.
        """
        # Multiply the source output by the current STW.
        # This means reinforcement can change future
        # signal strengths without changing this method.
        signal = neuron_output * self.stw

        if self.conn_type == 0:
            return signal, 0.0
        else:
            return 0.0, signal        

    def reinforce(self, reward, punishment, learning_rate=0.5):
        """Adjust the short-term weight using the relevant chemical signal.

        Args:
            reward (float): Scalar reinforcement for excitatory connections,
                e.g. 0.4; ignored for inhibitory connections.
            punishment (float): Scalar reinforcement for inhibitory connections,
                e.g. 0.3; ignored for excitatory connections.
            learning_rate (float): Scalar multiplier for susceptibility and the
                selected chemical signal; default 0.5.

        Returns:
            None. Updates self.stw in place and clamps it to [0.0, 1.0].
            Leaves self.ltw and self.susceptibility unchanged.
        """
        chemical = reward
        if self.conn_type == 1:
            chemical = punishment
        
        change = learning_rate * self.susceptibility * chemical
        self.stw = max(0.0, min(self.stw + change, 1.0))
        
    def settle_weights(self, stw_rate=0.2, ltw_rate=0.05):
        """Move short-term and long-term weights toward each other.

        Args:
            stw_rate (float): Fraction of the old ltw - stw gap applied to stw;
                default 0.2.
            ltw_rate (float): Fraction of the old stw - ltw gap applied to ltw;
                default 0.05. Both rates normally lie in [0.0, 1.0], but are
                not validated.

        Returns:
            None. Updates self.stw and self.ltw using their previous values.
            For stw=0.8 and ltw=0.2, the defaults produce approximately 0.68
            and 0.23 respectively. No clamping is applied.
        """
        
        new_stw = self.stw + stw_rate * (self.ltw - self.stw)
        new_ltw = self.ltw + ltw_rate * (self.stw - self.ltw)
        
        # We need to do this approach as both calculations rely on the previous values
        self.stw = new_stw
        self.ltw = new_ltw
