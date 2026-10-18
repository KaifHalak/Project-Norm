""" Mechanism where chemical is released in response to an action / condition which changes the concentration of the chemicals in the body """
    

class Emitter:
    def __init__(self, chemical, amount):
        """
        chemical: Name of chemical to release.
        amount: How much to release when activated.

        Example:
        Emitter("hunger", 0.05)
        """

        self.chemical = chemical
        self.amount = amount

    def emit(self, body, active=True):
        """
        Release a chemical when the emitter is active.

        body: Biochemistry object
        active: Whether the emitter is activated

        Example:
        If hunger = 0.4 and amount = 0.05,
        hunger should become 0.45.
        """
        
        if not active:
            return
        
        body.add_chemical(self.chemical, self.amount)

        
