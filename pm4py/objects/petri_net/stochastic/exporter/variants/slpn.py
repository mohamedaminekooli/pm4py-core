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

from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet


def export_petri_to_spn(StochasticPetriNet: StochasticPetriNet, im: Marking, output_filename):
    """
    Export a StochasticPetriNet to an SLPN file

    Parameters
    ----------
    StochasticPetriNet: :class:`pm4py.entities.petri.petrinet.stochastic.StochasticPetriNet`
        StochasticPetriNet
    im: :class:`pm4py.entities.petri.petrinet.Marking`
        Marking
    output_filename:
        Absolute output file name for saving the slpn file
    """
    num_places = len(StochasticPetriNet.places)
    num_transitions = len(StochasticPetriNet.transitions)

    with open(output_filename, "w") as file:
        # Write the number of places
        file.write(f"# number of places\n{num_places}\n")
        # Write the initial marking
        file.write(f"# initial marking\n")
        places_sort_list_im = sorted([x for x in list(StochasticPetriNet.places) if x in im], key=lambda x: x.name)
        places_sort_list_not_im = sorted(
        [x for x in list(StochasticPetriNet.places) if x not in im], key=lambda x: x.name)
        sorted_places = places_sort_list_im+places_sort_list_not_im
        for place in sorted_places:
            marking_value = im[place] if place in im else 0
            file.write(f"{marking_value}\n")
        # Write the number of transitions
        file.write(f"# number of transitions\n{num_transitions}\n")
        # Write information for each transition
        tran_num = 0
        for transition in StochasticPetriNet.transitions:
            file.write(f"# transition {tran_num}\n")
            # Write the label (silent or actual label)
            if transition.label is not None:
                file.write(f"label {transition.label}\n")
            else:
                file.write("silent\n")
            # Write the weight
            file.write(f"# weight\n{transition.weight}\n")
            # Write the number of input places
            file.write(f"# number of input places\n{len(transition.in_arcs)}\n")
            # Write the input places
            for arc in transition.in_arcs:
                index_place = sorted_places.index(arc.source)
                file.write(f"{index_place}\n")
            # Write the number of output places
            file.write(f"# number of output places\n{len(transition.out_arcs)}\n")
            # Write the output places
            for arc in transition.out_arcs:
                index_place = sorted_places.index(arc.target)
                file.write(f"{index_place}\n")
            tran_num += 1
