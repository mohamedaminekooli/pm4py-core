from pm4py.objects.petri_net.stochastic.weightestimators.abstractfrequencyestimator import view_stochastic_petri_net
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to_spn

def import_spn(file_path):
    """
    Import a Stochastic Petri net from an SLPN file

    Parameters
    ----------
    input_file_path
        Input file path

    Returns
    -----------
    net
        Petri net
    im
        Initial marking
    """
    spn = StochasticPetriNet()
    im = Marking()
    with open(file_path, 'r') as file:
        lines = file.readlines()
        idx = 0
        number_places = 0
        number_transitions = 0
        j = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith("#"):
                idx += 1
                continue
            if line.startswith("label") or line.startswith("silent"):
                name = f"t{j}"
                label = line.split(" ")[1] if line.startswith("label") else None
                transition = StochasticPetriNet.Transition(name, label)
                spn.transitions.add(transition)
                idx += 1
                while lines[idx].startswith("#"):
                    idx += 1
                transition.weight = float(lines[idx])
                idx += 1
                while lines[idx].startswith("#"):
                    idx += 1
                num_input_places = int(lines[idx])
                for _ in range(num_input_places):
                    idx += 1
                    place_id = lines[idx].strip()
                    for place in spn.places:
                        if place_id == place.name:
                            source = place
                            break
                    target = transition
                    arc = add_arc_from_to_spn(source, target, spn)
                    spn.arcs.add(arc)
                    #transition.in_arcs.add(place_name)
                idx += 1
                while lines[idx].startswith("#"):
                    idx += 1
                num_output_places = int(lines[idx])
                for _ in range(num_output_places):
                    idx += 1
                    place_id = lines[idx].strip()
                    for place in spn.places:
                        if place_id == place.name:
                            target = place
                            break
                    if not target in spn.places:
                        spn.places.add(target)
                    source = transition
                    arc = add_arc_from_to_spn(source, target, spn)
                    spn.arcs.add(arc)
                idx += 1
                j += 1
            elif not number_places:
                number_places = int(line)
                idx += 1
                while lines[idx].startswith("#"):
                    idx += 1
                for i in range(number_places):
                    place_id = str(i)
                    place = StochasticPetriNet.Place(place_id)
                    spn.places.add(place)
                    if int((lines[idx].strip())):
                        im[place] = int(lines[idx].strip())
                    idx += 1
            elif not number_transitions:
                number_transitions = int(line)
                idx += 1
    return spn, im
import os
def import_spln_script():
    spn, im = import_spn(os.path.join("..", "tests", "input_data", "PetriNet.slpn"))
    view_stochastic_petri_net(spn, im, format="svg")

    for place in spn.places:
        print(place)
    for transition in spn.transitions:
        print(transition.name, transition.label, transition.weight)
        print(transition.in_arcs)
        for output_place in transition.out_arcs:
            print(output_place, transition.name)
    print(f"im: {im}")

import_spln_script()