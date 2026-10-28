
""" Genetic instructions for a Norn. """

import random

class Genome:
    def __init__(self, genes=None):
        """
        A genome stores the inherited instructions
        used to construct a Norn.

        genes:
            Optional dictionary containing genetic values.

        If no genes are supplied, use the default genome.
        """

        default_genes = {
            "neuron_threshold": 0.0,
            "relaxation_rate": 1.0,
            "learning_rate": 0.5,

            "eat_exc_weight": 0.4,
            "eat_inh_weight": 0.1,
            "ignore_weight": 0.25,

            "hunger_rate": 0.05,
            "pain_decay": 0.9
        }

        self.genes = (default_genes if genes is None else genes).copy()

    def get_gene(self, name):
        """
        Return the value of a particular gene.

        Example:
        genome.get_gene("learning_rate") -> 0.5
        """

        return self.genes[name]

    def set_gene(self, name, value):
        """
        Update a gene's value.

        For now, allow changes only to existing genes.
        We will add mutation rules and gene-specific
        limits in Phase 9D.
        """

        if name not in self.genes:
            raise KeyError(name)
        self.genes[name] = value
        
    
    def reproduce(self, partner):
        """
        Create a new genome by combining genes
        from two parent genomes.

        self: Parent A's genome.
        partner: Parent B's genome.

        Each gene has a 50% chance of being
        inherited from either parent. 
        
        This is called uniform crossover.
        """

        # STEP 1: Check that both parents have
        # the same set of genes.
        if self.genes.keys() != partner.genes.keys():
            raise ValueError("Parents must have matching genes.")

        # STEP 2: Create an empty dictionary
        # to store the child's genetic instructions.
        child_genes = {}

        # STEP 3: Go through every gene.
        for gene_name in self.genes:

            # Read the gene from both parents.
            parent_a_value = self.genes[gene_name]
            parent_b_value = partner.genes[gene_name]

            # Randomly select which parent's value
            # the child will inherit.
            inherited_value = random.choice([
                parent_a_value,
                parent_b_value
            ])

            # Store the inherited gene.
            child_genes[gene_name] = inherited_value

        # STEP 4: Construct a NEW genome object.
        # This genome is independent of the parents.
        child = Genome(child_genes)

        return child
    
    
def mutate(self, mutation_rate=0.1):
    """
    Randomly mutate genes in this genome.

    mutation_rate:
        Probability that each gene will mutate.

    Mutation:
        Adds a small random positive or negative
        value to the selected gene.

    Returns:
        Dictionary describing which genes changed.
    """

    # STEP 1: Validate the mutation probability.
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            "Mutation rate must be between 0 and 1."
        )

    # STEP 2: Define mutation rules for each gene.
    #
    # Each tuple contains:
    # (minimum_value for gene, maximum_value for gene, maximum_change)
    #
    # Gene-specific limits prevent invalid values.
    mutation_rules = {
        "neuron_threshold": (0.0, 1.0, 0.05),
        "relaxation_rate": (0.0, 1.0, 0.05),
        "learning_rate": (0.0, 1.0, 0.05),

        "eat_exc_weight": (0.0, 1.0, 0.05),
        "eat_inh_weight": (0.0, 1.0, 0.05),
        "ignore_weight": (0.0, 1.0, 0.05),

        "hunger_rate": (0.0, 0.2, 0.01),
        "pain_decay": (0.0, 1.0, 0.05)
    }

    # Record mutations for inspection.
    mutations = {}

    # STEP 3: Examine every gene.
    for gene_name, old_value in self.genes.items():

        # Randomly decide whether this gene mutates.
        if random.random() >= mutation_rate:
            continue

        # Retrieve the gene's permitted limits.
        min_value, max_value, max_change = (
            mutation_rules[gene_name]
        )

        # STEP 4: Generate a random mutation.
        #
        # It can increase or decrease the gene value.
        change = random.uniform(
            -max_change,
            max_change
        )

        new_value = old_value + change

        # STEP 5: Clamp the mutation within
        # the gene's valid range.
        new_value = max(
            min_value,
            min(new_value, max_value)
        )

        # STEP 6: Store the mutated gene.
        self.genes[gene_name] = new_value

        # Record the mutation if the value changed.
        if new_value != old_value:
            mutations[gene_name] = (
                old_value,
                new_value
            )

    return mutations


