import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1.0
    
    for person in people:
        # Determine number of gene copies for this person
        if person in two_genes:
            gene_count = 2
        elif person in one_gene:
            gene_count = 1
        else:
            gene_count = 0
        
        # Determine if person has trait
        has_trait = person in have_trait
        
        # Calculate probability for this person
        person_prob = 1.0
        
        # Check if person has parents in the data
        mother = people[person]["mother"]
        father = people[person]["father"]
        
        if mother is None and father is None:
            # No parents: use unconditional probability
            person_prob *= PROBS["gene"][gene_count]
        else:
            # Has parents: calculate probability of inheriting gene_count
            # Calculate probability for each parent passing on the gene
            
            # Function to compute probability of inheriting a gene from a parent
            def inherit_prob(parent, child_gene_count):
                # Determine parent's gene count
                if parent in two_genes:
                    parent_gene = 2
                elif parent in one_gene:
                    parent_gene = 1
                else:
                    parent_gene = 0
                
                # Probability of passing the gene (without mutation)
                if parent_gene == 2:
                    pass_prob = 1.0
                elif parent_gene == 1:
                    pass_prob = 0.5
                else:  # parent_gene == 0
                    pass_prob = 0.0
                
                # With mutation
                # Pass gene if: (passes without mutation) OR (doesn't pass but mutates)
                # Or for child's total count, we need to consider combinations
                # Return probability of passing exactly 0 or 1 gene from this parent
                prob_pass = pass_prob * (1 - PROBS["mutation"]) + (1 - pass_prob) * PROBS["mutation"]
                prob_not_pass = 1 - prob_pass
                
                if child_gene_count == 1:
                    # This parent contributes exactly 1 gene
                    return prob_pass
                else:
                    # This parent contributes 0 genes
                    return prob_not_pass
            
            # Get probabilities from mother and father
            # Child can have 0, 1, or 2 genes based on inheritance from both parents
            if gene_count == 0:
                # Must get 0 from mother AND 0 from father
                mother_prob = inherit_prob(mother, 0)
                father_prob = inherit_prob(father, 0)
                person_prob *= mother_prob * father_prob
            elif gene_count == 2:
                # Must get 1 from mother AND 1 from father
                mother_prob = inherit_prob(mother, 1)
                father_prob = inherit_prob(father, 1)
                person_prob *= mother_prob * father_prob
            else:  # gene_count == 1
                # Either: 1 from mother and 0 from father, OR 0 from mother and 1 from father
                prob1 = inherit_prob(mother, 1) * inherit_prob(father, 0)
                prob2 = inherit_prob(mother, 0) * inherit_prob(father, 1)
                person_prob *= (prob1 + prob2)
        
        # Multiply by trait probability
        person_prob *= PROBS["trait"][gene_count][has_trait]
        
        # Multiply into joint probability
        joint_prob *= person_prob
    
    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Update gene distribution
        if person in two_genes:
            probabilities[person]["gene"][2] += p
        elif person in one_gene:
            probabilities[person]["gene"][1] += p
        else:
            probabilities[person]["gene"][0] += p
        
        # Update trait distribution
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # Normalize gene distribution
        gene_total = sum(probabilities[person]["gene"].values())
        if gene_total > 0:
            for gene_count in probabilities[person]["gene"]:
                probabilities[person]["gene"][gene_count] /= gene_total
        
        # Normalize trait distribution
        trait_total = sum(probabilities[person]["trait"].values())
        if trait_total > 0:
            for trait_value in probabilities[person]["trait"]:
                probabilities[person]["trait"][trait_value] /= trait_total


if __name__ == "__main__":
    main()
