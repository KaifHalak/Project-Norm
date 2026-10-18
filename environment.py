""" Environment module. It contains information which the norns can sense and interact with, but does not know if useful / harmful """

class Environment:
    def __init__(self):
        self.mushrooms = {
            "blue": {
                "hunger_change": -0.3,
                "pain_change": 0.0
            },
            "orange": {
                "hunger_change": 0.0,
                "pain_change": 0.4
            }
        }

    def execute_action(self, body, action, mushroom_color):

        # Ignoring a mushroom has no immediate effect.
        if action != "Eat":
            return

        # Retrieve the mushroom's properties.
        mushroom = self.mushrooms[mushroom_color]

        # Apply its effects to the Norn's body.
        body.add_chemical(
            "hunger",
            mushroom["hunger_change"]
        )

        body.add_chemical(
            "pain",
            mushroom["pain_change"]
        )
