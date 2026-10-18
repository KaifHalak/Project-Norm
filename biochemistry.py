""" Responsible for generating reward / punishment feedback """



class Biochemistry:
    def __init__(self):

        # Chemical concentrations range from 0.0 to 1.0
        self.chemicals = {
            "hunger": 0.8,
            "pain": 0.0,
            "reward": 0.0,
            "punishment": 0.0
        }

    def get_chemical(self, name):
        """
        Return the current concentration of a chemical.

        Example:
        get_chemical("hunger") -> 0.8
        """

        return self.chemicals.get(name, 0.0)  # Default to 0.0 if not found
        

    def add_chemical(self, name, amount):
        """
        Increase or decrease a chemical concentration.

        Example:
        Hunger = 0.8
        add_chemical("hunger", -0.3)
        New hunger = 0.5

        IMPORTANT:
        Keep concentrations between 0.0 and 1.0.
        """


        curVal = self.chemicals.get(name, 0.0)
        newVal = max(0.0, min(curVal + amount, 1.0))
        self.chemicals[name] = newVal

        

    def decay_chemical(self, name, factor=0.9):
        """
        Gradually decrease a chemical concentration.

        Example:
        Pain = 0.5
        decay_chemical("pain", 0.9)
        New pain = 0.45
        """
        curVal = self.chemicals.get(name, 0.0)
        newVal = curVal * factor
        self.chemicals[name] = newVal


    
    def react(self, old_hunger, old_pain):
        # Reward when hunger decreases.
        self.chemicals["reward"] = max(
            0.0,
            old_hunger - self.chemicals["hunger"]
        )

        # Punishment when pain increases.
        self.chemicals["punishment"] = max(
            0.0,
            self.chemicals["pain"] - old_pain
        )

