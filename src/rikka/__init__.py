import sys

from .particle_filter import run_particle_filter
from .pdr import run


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "particle":
        run(use_particle_filter=True)
    else:
        run()


__all__ = ["run", "run_particle_filter"]
