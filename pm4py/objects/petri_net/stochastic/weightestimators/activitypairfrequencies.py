from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from enum import Enum
from pm4py.objects.log.obj import EventLog
from pm4py.util import constants, exec_utils, xes_constants as xes
from typing import Optional, Dict, Any
from pm4py.statistics.attributes.log import get as log_attributes

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY

class FrequencyCalculator:
    def __init__(self, log: EventLog, parameters: Optional[Dict[Any, Any]] = None):
        if parameters is None:
            parameters = {}
        self.log = log
        self.parameters = parameters

    def scan_log(self):
        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, self.parameters, xes.DEFAULT_NAME_KEY)
        activities_occurrences = log_attributes.get_attribute_values(self.log, activity_key, parameters=self.parameters)
        # activities = list(activities_occurrences.keys())
        return activities_occurrences

    def calculate_follows_frequency(self):
        activities_occurrences = self.scan_log()
        follows_frequency = {}  # Assuming a dictionary structure to store follows frequency

        for case in self.log:
            activities = [event[Parameters.ACTIVITY_KEY] for event in case]
            for i in range(len(activities) - 1):
                current_activity = activities[i]
                next_activity = activities[i + 1]

                if current_activity not in follows_frequency:
                    follows_frequency[current_activity] = {}

                if next_activity not in follows_frequency[current_activity]:
                    follows_frequency[current_activity][next_activity] = 1
                else:
                    follows_frequency[current_activity][next_activity] += 1

        return follows_frequency

    def calculate_start_frequency(self):
        activities_occurrences = self.scan_log()
        start_frequency = {}  # Assuming a dictionary structure to store start frequency

        for case in self.log:
            start_activity = case[0][Parameters.ACTIVITY_KEY]
            if start_activity not in start_frequency:
                start_frequency[start_activity] = 1
            else:
                start_frequency[start_activity] += 1

        return start_frequency

    def calculate_end_frequency(self):
        activities_occurrences = self.scan_log()
        end_frequency = {}  # Assuming a dictionary structure to store end frequency

        for case in self.log:
            end_activity = case[-1][Parameters.ACTIVITY_KEY]
            if end_activity not in end_frequency:
                end_frequency[end_activity] = 1
            else:
                end_frequency[end_activity] += 1

        return end_frequency

# Example usage:
# Assuming you have an EventLog named 'event_log'
# and the necessary frequencies and mappings
#follows_frequency_matrix = FrequencyCalculator(event_log).calculate_follows_frequency()
#start_frequency_column = FrequencyCalculator(event_log).calculate_start_frequency()
#end_frequency_column = FrequencyCalculator(event_log).calculate_end_frequency()

# Create an instance of the LH estimator
#lh_estimator = ActivityPairLHWeightEstimator(follows_frequency_matrix, start_frequency_column, end_frequency_column)

# Create an instance of the RH estimator
#rh_estimator = ActivityPairRHWeightEstimator(follows_frequency_matrix, start_frequency_column, end_frequency_column)

# Assuming you have a StochasticPetriNet named 'spn'
# Estimate weights using LH estimator
#lh_estimator.estimate_weights(spn, FrequencyCalculator(event_log))

# Assuming you have another StochasticPetriNet named 'spn2'
# Estimate weights using RH estimator
#rh_estimator.estimate_weights(spn2, FrequencyCalculator(event_log))
