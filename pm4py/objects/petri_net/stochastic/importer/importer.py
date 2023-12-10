'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
from enum import Enum

from pm4py.objects.petri_net.stochastic.importer.variants import slpn
from pm4py.util import exec_utils


class Variants(Enum):
    SLPN = slpn


SLPN = Variants.SLPN


def apply(input_file_path, variant=SLPN, parameters=None):
    """
    Import a Petri net from a PNML file

    Parameters
    ------------
    input_file_path
        Input file path
    parameters
        Other parameters of the importer
    variant
        Variant of the algorithm to use, possible values:
            - Variants.PNML
    """
    return exec_utils.get_variant(variant).import_slpn(input_file_path, parameters=parameters)
