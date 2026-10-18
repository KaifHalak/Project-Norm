
class Biochemistry:
    def __init__(self, hunger=0.8):
        """Initialize the example's chemical concentrations.

        Args:
            hunger (float): Initial scalar hunger concentration, e.g. 0.8;
                default 0.8. Expected to be nonnegative; not validated.

        Returns:
            None. Initializes self.chemicals as dict[str, float] with keys
            "hunger", "pain", "hunger_decrease", "pain_increase", "reward",
            and "punishment". All concentrations except hunger start at 0.0.
        """
        # Simulated chemical concentrations.
        self.chemicals = {
            "hunger": hunger,
            "pain": 0.0,
            "hunger_decrease": 0.0,
            "pain_increase": 0.0,
            "reward": 0.0,
            "punishment": 0.0
        }

    def emit(self, action):
        """Introduce chemicals associated with an environmental action.

        Args:
            action (str): "eat_safe" adds 0.4 hunger_decrease; "eat_poison"
                adds 0.4 hunger_decrease and 0.3 pain_increase. Other values
                leave concentrations unchanged.

        Returns:
            None. Adds the corresponding amounts to self.chemicals in place.
        """
        # Object interactions introduce chemicals.
        if action == "eat_safe":
            self.chemicals["hunger_decrease"] += 0.4

        elif action == "eat_poison":
            # Poisonous food may reduce hunger
            # while also increasing pain.
            self.chemicals["hunger_decrease"] += 0.4
            self.chemicals["pain_increase"] += 0.3

    def react(self):
        """Convert available hunger reduction and pain increase into signals.

        Args:
            No explicit parameters; reads the scalar values in self.chemicals.

        Returns:
            None. Consumes min(hunger, hunger_decrease) from both concentrations
            and adds that amount to reward. Transfers all pain_increase into
            pain and punishment, then resets pain_increase to 0.0.
        """
        c = self.chemicals

        # Reaction 1:
        # Hunger Decrease + Hunger -> Reward
        amount = min(c["hunger"], c["hunger_decrease"])

        c["hunger"] -= amount
        c["hunger_decrease"] -= amount
        c["reward"] += amount

        # Reaction 2:
        # Pain Increase -> Pain + Punishment
        amount = c["pain_increase"]

        c["pain"] += amount
        c["punishment"] += amount
        c["pain_increase"] = 0.0

    def read_receptors(self):
        """Read the current reinforcement signals.

        Args:
            No explicit parameters; reads self.chemicals.

        Returns:
            tuple[float, float]: (reward, punishment), e.g. (0.4, 0.3) after
            eating poison with initial hunger=0.8 and running react once.
            Reading does not reset either concentration.
        """
        # Simplified receptors expose chemical signals
        # to the neural learning system.
        c = self.chemicals
        return c["reward"], c["punishment"]


def simulate(hunger, action):
    """Run and print one biochemical reinforcement example.

    Args:
        hunger (float): Initial scalar hunger concentration, e.g. 0.8.
        action (str): Environmental action, e.g. "eat_safe" or "eat_poison".
            Other values introduce no chemicals.

    Returns:
        None. Prints the action, initial and final hunger, reward, punishment,
        and resulting excitatory and inhibitory short-term weights. All
        simulation objects and scalar weights are local to this call.
    """
    body = Biochemistry(hunger=hunger)

    # The environment introduces chemicals.
    body.emit(action)

    # The biochemical engine runs its reactions.
    body.react()

    # Receptors supply signals to the neural network.
    reward, punishment = body.read_receptors()

    # Existing eligible connections respond.
    eligibility = 0.6
    learning_rate = 0.5

    excitatory_stw = 0.2
    inhibitory_stw = 0.1

    excitatory_stw += learning_rate * eligibility * reward
    inhibitory_stw += learning_rate * eligibility * punishment

    print(f"Action: {action}, initial hunger: {hunger}")
    print(f"Final hunger: {body.chemicals['hunger']:.2f}")
    print(f"Reward: {reward:.2f}, Punishment: {punishment:.2f}")
    print(f"Excitatory STW: {excitatory_stw:.2f}")
    print(f"Inhibitory STW: {inhibitory_stw:.2f}")
    print()


# Scenario 1: Hungry and eats safe food.
simulate(hunger=0.8, action="eat_safe")

# Scenario 2: Full and eats safe food.
simulate(hunger=0.0, action="eat_safe")

# Scenario 3: Hungry and eats poisonous food.
simulate(hunger=0.8, action="eat_poison")
