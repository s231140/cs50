import sys

from nim import train, play


def main():
    """Train a Nim AI and play a game against it."""
    if len(sys.argv) > 2:
        sys.exit("Usage: python play.py [games]")

    games = int(sys.argv[1]) if len(sys.argv) == 2 else 10000
    ai = train(games)
    play(ai)


if __name__ == "__main__":
    main()

