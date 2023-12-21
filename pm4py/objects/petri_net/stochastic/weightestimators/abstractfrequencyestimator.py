from collections import defaultdict
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.log.obj import EventLog
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.util import constants, exec_utils, xes_constants as xes
from typing import Optional, Dict, Any
from pm4py.objects.petri_net.obj import PetriNet
from enum import Enum

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY

class AbstractFrequencyEstimator:
    def __init__(self):
        # Initialize a dictionary to store activity frequencies
        self.activity_frequency = defaultdict(float)

    # Calculate transition weights based on event frequencies
    def estimate_weights_apply(self, log: EventLog, pn: PetriNet, parameters: Optional[Dict[Any, Any]] = None):
        self.activity_frequency = self.scan_log(log, pn, parameters)
        spn = StochasticPetriNet(pn)
        return self.estimate_weights_activity_frequencies(spn)
    
    def scan_log(self, log: EventLog, pn: PetriNet, parameters: Optional[Dict[Any, Any]] = None):
        if parameters is None:
            parameters = {}

        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes.DEFAULT_NAME_KEY)
        activities_occurrences = log_attributes.get_attribute_values(log, activity_key, parameters=parameters)
        for transition in pn.transitions:
            if transition.label not in activities_occurrences:
                activities_occurrences[transition.label] = 1
        return activities_occurrences

    # Assign weights to transitions based on event frequencies
    def estimate_weights_activity_frequencies(self, spn: StochasticPetriNet):
        for transition in spn.transitions:
            weight = self.load_activity_frequency(transition)
            transition.weight = weight
        return spn

    # Retrieve the frequency of a specific activity
    def load_activity_frequency(self, tran):
        activity = tran.label
        # Use a default value of 0.0 if the activity is not found in the log
        frequency = float(self.activity_frequency.get(activity, 0.0))
        return frequency

