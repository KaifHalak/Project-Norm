
class Biochemistry:
    def __init__(self, hunger=0.8):
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
        # Object interactions introduce chemicals.
        if action == "eat_safe":
            self.chemicals["hunger_decrease"] += 0.4

        elif action == "eat_poison":
            # Poisonous food may reduce hunger
            # while also increasing pain.
            self.chemicals["hunger_decrease"] += 0.4
            self.chemicals["pain_increase"] += 0.3

    def react(self):
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
        # Simplified receptors expose chemical signals
        # to the neural learning system.
        c = self.chemicals
        return c["reward"], c["punishment"]


def simulate(hunger, action):
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
