from enum import Enum

from pm4py.objects.petri_net.stochastic.importer.variants import slpn
from pm4py.util import exec_utils


class Variants(Enum):
    SLPN = slpn


SLPN = Variants.SLPN


def apply(input_file_path, variant=SLPN):
    """
    Import a Petri net from a PNML file

    Parameters
    ------------
    input_file_path
        Input file path
    variant
        Variant of the algorithm to use, possible values:
            - Variants.SLPN
    """
    return exec_utils.get_variant(variant).import_spn(input_file_path)
