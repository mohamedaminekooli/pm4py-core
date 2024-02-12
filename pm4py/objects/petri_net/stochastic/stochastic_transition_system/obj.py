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
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.objects.transition_system import constants


class StochasticTransitionSystem(TransitionSystem):
    def __init__(self, ts: Optional[TransitionSystem] = None):
        """
        Constructor method for StochasticPetriNet.

        Parameters:
        - petri_net (Optional[PetriNet]): An optional existing PetriNet instance to initialize the StochasticPetriNet.

        """
        if ts:
            super().__init__(name=ts.name, states=ts.states, transitions=ts.transitions)
        else:
            super().__init__()

    class Transition(TransitionSystem.Transition):

        def __init__(self, name: str, from_state: str = Collection[TransitionSystem.State], to_state: Collection[TransitionSystem.State] = None, data:Dict[str, Any] = None, weight: float = 1.0):
            super().__init__(name, from_state, to_state, data)
            self.__weight = weight

        def __set_weight(self, weight: float):
            self.__weight = weight

        def __get_weight(self) -> float:
            return self.__weight

        weight = property(__get_weight, __set_weight)
