from dataclasses import dataclass

@dataclass
class Scaffold:
    """Template to use, only change specific positions on reference sequence"""
    base: str                   # base helms sequence, e.g. 'P.E.P.T.I.D.E' in the linear peptide
    max_n_start: int            # max number of tokens to change at the beginning
    start_prob: float = 0.5     # probability to generate start the base peptide
    cyclic_prob: float = 0.0    # probability to generate end2end cyclic peptides
