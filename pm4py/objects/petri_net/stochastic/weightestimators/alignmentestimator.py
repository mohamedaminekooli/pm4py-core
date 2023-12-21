from pm4py.objects.petri_net.obj import PetriNet
from collections import defaultdict
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.log.obj import EventLog
from pm4py.util import constants
from pm4py.objects.petri_net.obj import PetriNet
from enum import Enum
from pm4py.objects.petri_net.stochastic.utils import align_utils
from pm4py.objects.conversion.log import converter

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY

class Alignmentestimator:
    def __init__(self, log, net, im, fm):
        # Initialize a dictionary to store activity frequencies
        self.activity_weights = defaultdict(float)
        self.log=converter.apply(log, variant=converter.Variants.TO_EVENT_LOG)
        self.net=net
        self.im=im
        self.fm=fm

    def align(self, trace, net, im, fm):
        return align_utils.apply(trace, net, im, fm)

    def walign(self):
        alignments = []
        silents_occurencies = {}
        for trace in self.log:
            alignment= self.align(trace, self.net, self.im, self.fm)
            alignments.append(alignment)
        for alignment in alignments:
            for transition, occurency in alignment['silent_occurence'].items():
                silents_occurencies[transition] = silents_occurencies.get(transition, 0.0) + occurency
        #pretty_print_alignments(alignments)
        walign={}
        # Count occurrences of the transition in the alignments
        for transition in self.net.transitions:
            if transition.label is not None:
                walign[transition] = sum(1 for alignment in alignments for event in alignment['alignment'] if event[1] == transition.label)
            else:
                walign[transition] = silents_occurencies[transition.name]

        return walign
    # Calculate transition weights based on event frequencies
    def estimate_weights_apply(self, log: EventLog, pn: PetriNet):
        self.activity_weights = self.walign()
        spn = StochasticPetriNet(pn)
        return self.estimate_weights_activity_frequencies(spn)

    # Assign weights to transitions based on event frequencies
    def estimate_weights_activity_frequencies(self, spn: StochasticPetriNet):
        for transition in spn.transitions:
            weight = self.load_activity_frequency(transition)
            transition.weight = weight
        return spn

    # Retrieve the frequency of a specific activity
    def load_activity_frequency(self, tran):
        activity = tran
        # Use a default value of 0.0 if the activity is not found in the log
        frequency = float(self.activity_weights.get(activity, 0.0))
        return frequency