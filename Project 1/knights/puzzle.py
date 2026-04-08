from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

ASaidKnight = Symbol("A said I am a knight")
ASaidKnave = Symbol("A said I am a knave")


def one_of(a, b):
    return And(Or(a, b), Not(And(a, b)))


def speaker_statement(knight, knave, statement):
    return And(
        Biconditional(knight, statement),
        Biconditional(knave, Not(statement))
    )


# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    one_of(AKnight, AKnave),
    speaker_statement(AKnight, AKnave, And(AKnight, AKnave))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    one_of(AKnight, AKnave),
    one_of(BKnight, BKnave),
    speaker_statement(AKnight, AKnave, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    one_of(AKnight, AKnave),
    one_of(BKnight, BKnave),
    speaker_statement(AKnight, AKnave, Biconditional(AKnight, BKnight)),
    speaker_statement(BKnight, BKnave, Not(Biconditional(AKnight, BKnight)))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    one_of(AKnight, AKnave),
    one_of(BKnight, BKnave),
    one_of(CKnight, CKnave),
    one_of(ASaidKnight, ASaidKnave),
    Or(
        And(
            ASaidKnight,
            speaker_statement(AKnight, AKnave, AKnight)
        ),
        And(
            ASaidKnave,
            speaker_statement(AKnight, AKnave, AKnave)
        )
    ),
    speaker_statement(BKnight, BKnave, ASaidKnave),
    speaker_statement(BKnight, BKnave, CKnave),
    speaker_statement(CKnight, CKnave, AKnight)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
