from enum import Enum
from pm4py.objects.petri_net.stochastic.exporter.variants import slpn
from pm4py.util import exec_utils


class Variants(Enum):
    SLPN = slpn

SLPN = Variants.SLPN

def apply(spn, initial_marking, output_filename, variant=SLPN):
    """
    Export a Stochastic Petri net along with an initial marking to an output file

    Parameters
    ------------
    spn
        Stochastic Petri net
    initial_marking
        Initial marking
    output_filename
        Output filename
    variant
        Variant of the algorithm, possible values:
            - Variants.SLPN
    """
    return exec_utils.get_variant(variant).export_petri_to_spn(spn, initial_marking, output_filename)

