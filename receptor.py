""" Mechanism where a chemical is detected, the concentrations are changed, and triggers a response in the body. """


class Receptor:
    def __init__(self, chemical, threshold):
        """
        chemical: The chemical to detect.
        threshold: Minimum concentration required
                   to activate the receptor.

        Example:
        Receptor("hunger", 0.5)
        """

        self.chemical = chemical
        self.threshold = threshold

    def is_active(self, body):
        """
        Check whether the chemical concentration
        has reached the receptor's threshold.

        Examples:
        Hunger = 0.8, threshold = 0.5 -> True
        Hunger = 0.3, threshold = 0.5 -> False
        """

    
        return body.get_chemical(self.chemical) >= self.threshold
