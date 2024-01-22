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

from typing import Any, Collection, Dict, Optional

from pm4py.objects.petri_net.obj import PetriNet


class StochasticPetriNet(PetriNet):
    """
    Represents a Stochastic Petri Net, a subclass of the standard Petri Net.

    This class extends the PetriNet class to include additional functionality
    related to stochastic aspects.

    Attributes:
    - name (str): The name of the Stochastic Petri Net.
    - places (Collection[Place]): Collection of places in the Petri Net.
    - transitions (Collection[Transition]): Collection of transitions in the Petri Net.
    - arcs (Collection[Arc]): Collection of arcs connecting places and transitions.

    Methods:
    - __init__(petri_net: Optional[PetriNet] = None): Constructor method. Initializes a StochasticPetriNet instance.
    - Transition: Nested class representing transitions in the Stochastic Petri Net.

    """
    def __init__(self, petri_net: Optional[PetriNet] = None):
        """
        Constructor method for StochasticPetriNet.

        Parameters:
        - petri_net (Optional[PetriNet]): An optional existing PetriNet instance to initialize the StochasticPetriNet.

        """
        if petri_net:
            super().__init__(name=petri_net.name, places=petri_net.places, transitions=petri_net.transitions, arcs=petri_net.arcs, properties=petri_net.properties)
        else:
            super().__init__()

    class Transition(PetriNet.Transition):
        """
        Represents a transition in the Stochastic Petri Net.

        This class extends the Transition class from the PetriNet to include
        additional attributes related to stochastic behavior.

        Attributes:
        - name (str): The name of the transition.
        - label (str): The label associated with the transition.
        - in_arcs (Collection[Arc]): Collection of arcs incoming to the transition.
        - out_arcs (Collection[Arc]): Collection of arcs outgoing from the transition.
        - weight (float): The weight or probability associated with the transition.
        - properties (Dict[str, Any]): Additional properties of the transition.

        Methods:
        - __init__(name: str, label: str = None, in_arcs: Collection[Arc] = None, out_arcs: Collection[Arc] = None,
                   weight: float = 1.0, properties: Dict[str, Any] = None): Constructor method for Transition.

        """
        def __init__(self, name: str, label: str = None, in_arcs: Collection[PetriNet.Arc] = None, out_arcs: Collection[PetriNet.Arc] = None, weight: float = 1.0, properties: Dict[str, Any] = None):
            super().__init__(name, label, in_arcs, out_arcs, properties)
            self.__weight = weight

        def __set_weight(self, weight: float):
            self.__weight = weight

        def __get_weight(self) -> float:
            return self.__weight

        weight = property(__get_weight, __set_weight)
