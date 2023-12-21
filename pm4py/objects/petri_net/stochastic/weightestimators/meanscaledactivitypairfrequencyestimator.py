from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from enum import Enum
from pm4py.objects.log.obj import EventLog
from pm4py.util import constants, exec_utils, xes_constants as xes
from typing import Optional, Dict, Any
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.statistics.start_activities.pandas import get as start_activities_get
from pm4py.statistics.end_activities.pandas import get as end_activities_get
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.conversion.log import converter


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY

class FrequencyCalculator:
    def __init__(self, log: EventLog, petrinet: PetriNet, parameters: Optional[Dict[Any, Any]] = None):
        if parameters is None:
            parameters = {}
        self.log = log
        self.pn = petrinet
        self.parameters = parameters

    def calculate_follows_frequency(self, current_activity, end_activities):
        follows_frequency = {}
        eventlog = converter.apply(self.log, variant=converter.Variants.TO_EVENT_LOG, parameters=self.parameters)
        for trace in eventlog:
            activities = [event["concept:name"] for event in trace]
            for i in range(len(activities) - 1):
                if activities[i] == current_activity and activities[i + 1] in end_activities:
                    follows_frequency[current_activity] = follows_frequency.get(current_activity, 0) + 1
        return follows_frequency

    def calculate_start_frequency(self, current_activity):
        start_frequency = {}
        start_activities = start_activities_get.get_start_activities(self.log, parameters=self.parameters)
        if current_activity in start_activities:
            start_frequency[current_activity] = start_activities[current_activity]
        else:
            start_frequency[current_activity] = 0
        return start_frequency

    def calculate_end_frequency(self, current_activity):
        end_frequency = {}
        end_activities = end_activities_get.get_end_activities(self.log, parameters=self.parameters)
        if current_activity in end_activities:
            end_frequency[current_activity] = end_activities[current_activity]
        else:
            end_frequency[current_activity] = 0
        return end_frequency

    def calculate_wrhpair(self):
        activities_weights = {}
        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, self.parameters, xes.DEFAULT_NAME_KEY)
        activities_occurrences = log_attributes.get_attribute_values(self.log, activity_key, parameters=self.parameters)
        end_activities = end_activities_get.get_end_activities(self.log, parameters=self.parameters)
        activities = list(activities_occurrences.keys())
        events = sum(activities_occurrences.values())
        for current_activity in activities:
            start_frequency = self.calculate_start_frequency(current_activity)
            end_frequency = self.calculate_end_frequency(current_activity)
            follows_frequency = self.calculate_follows_frequency(current_activity,end_activities)
            total_follows_frequency = follows_frequency[current_activity] if current_activity in follows_frequency else 0
            wrhpair = start_frequency[current_activity] + end_frequency[current_activity] + total_follows_frequency
            pairscale = wrhpair / (events/self.calculate_transitions(self.pn))
            activities_weights[current_activity] = pairscale
        
        return activities_weights
    
    def calculate_transitions(self, pn: PetriNet):
        return len(pn.transitions)

class MeanScaledActivityPairFrequencyEstimator:
    def __init__(self):
        self.activities_weights = {}

    def estimate_weights_apply(self, log: EventLog, pn: PetriNet):
        frequency_calculator = FrequencyCalculator(log, pn)
        self.activities_weights = frequency_calculator.calculate_wrhpair()
        spn = StochasticPetriNet(pn)
        return self.estimate_activity_pair_weights(spn)

    # Assign weights to transitions based on event frequencies
    def estimate_activity_pair_weights(self, spn: StochasticPetriNet):
        for transition in spn.transitions:
            weight = self.load_activities_weights(transition)
            transition.weight = weight
        return spn

    # Retrieve the frequency of a specific activity
    def load_activities_weights(self, tran):
        activity = tran.label
        # Use a default value of 0.0 if the activity is not found in the log
        frequency = float(self.activities_weights.get(activity, 1))
        return frequency