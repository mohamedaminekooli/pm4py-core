from typing import Union

import pandas as pd
from pm4py.objects.petri_net.obj import Marking, PetriNet
from collections import defaultdict
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.log.obj import EventLog, EventStream, Trace
from pm4py.util import constants
from enum import Enum
from pm4py.objects.petri_net.stochastic.utils import align_utils

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY

class AlignmentEstimator:
    def __init__(self, log: Union[EventLog, pd.DataFrame, EventStream], 
                 net: PetriNet, im: Marking, fm: Marking):
        """
        Initializes the AlignmentEstimator object.

        Parameters:
        - log: EventLog - Input event log
        - net: PetriNet - Input Petri net
        - im: Initial marking of the Petri net
        - fm: Final marking of the Petri net
        Attributes:
        - activity_weights: defaultdict(float) - Dictionary to store activity weights
        """
        self.activity_weights = defaultdict(float)
        self.log = log
        self.net = net
        self.im = im
        self.fm = fm

    def align(self, trace: Trace, net: PetriNet, im: Marking, fm: Marking):
        """
        Aligns a given trace with a Petri net.

        Parameters:
        - trace: List[Event] - Input trace to align
        - net: PetriNet - Input Petri net
        - im: Initial marking of the Petri net
        - fm: Final marking of the Petri net

        Returns:
        - alignment: Dict - Alignment result
        """
        return align_utils.apply(trace, net, im, fm)

    def walign(self):
        """
        Aligns all traces in the event log with the Petri net and calculates transition weights.

        Returns:
        - walign: Dict - Dictionary with transitions and their corresponding weights
        """
        alignments = []
        silents_occurrences = {}
        
        # Align each trace in the log with the Petri net
        for trace in self.log:
            alignment = self.align(trace, self.net, self.im, self.fm)
            alignments.append(alignment)
        # Count occurrences of silent transitions in the alignments
        for alignment in alignments:
            for transition, occurrence in alignment['silent_occurrence'].items():
                silents_occurrences[transition] = silents_occurrences.get(transition, 0.0) + occurrence
        
        walign = {}
        
        for transition in self.net.transitions:
            if transition.label is not None:
                walign[transition] = sum(1 for alignment in alignments for event in alignment['alignment'] if event[1] == transition.label)
            else:
                walign[transition] = silents_occurrences.get(transition.name, 0.0)

        return walign

    def estimate_weights_apply(self, pn: PetriNet):
        """
        Estimates transition weights based on alignment estimator.

        Parameters:
        - log: EventLog - Input event log
        - pn: PetriNet - Input Petri net

        Returns:
        - spn: StochasticPetriNet - Stochastic Petri net with estimated transition weights
        """
        self.activity_weights = self.walign()
        spn = StochasticPetriNet(pn)
        return self.estimate_weights_activity_frequencies(spn)

    def estimate_weights_activity_frequencies(self, spn: StochasticPetriNet):
        """
        Assigns weights to transitions in a Stochastic Petri net based on alignment results.

        Parameters:
        - spn: StochasticPetriNet - Stochastic Petri net

        Returns:
        - spn: StochasticPetriNet - Stochastic Petri net with updated transition weights
        """
        for transition in spn.transitions:
            weight = self.load_activity_frequency(transition)
            transition.weight = weight
        return spn

    def load_activity_frequency(self, tran):
        """
        Retrieves the frequency of a specific transition.

        Parameters:
        - tran: Transition - Input transition object

        Returns:
        - frequency: float - Frequency of the transition
        """
        activity = tran
        # Use a default value of 0.0 if the activity is not found in activity_weights
        frequency = float(self.activity_weights.get(activity, 0.0))
        return frequency
