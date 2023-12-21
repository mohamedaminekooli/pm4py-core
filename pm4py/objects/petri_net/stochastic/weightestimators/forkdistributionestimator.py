from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from enum import Enum
from pm4py.objects.log.obj import EventLog
from pm4py.util import constants, exec_utils, xes_constants as xes
from typing import Optional, Dict, Any
from pm4py.statistics.attributes.log import get as log_attributes
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
        self.parameters = parameters
        self.log = converter.apply(log, variant=converter.Variants.TO_EVENT_LOG, parameters=self.parameters)
        self.pn = petrinet
        self.activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, self.parameters, xes.DEFAULT_NAME_KEY)
    
    def scan_log(self, pn: PetriNet):
        activities_occurrences = log_attributes.get_attribute_values(self.log, self.activity_key, parameters=self.parameters)
        for transition in pn.transitions:
            if transition.label not in activities_occurrences:
                activities_occurrences[transition.label] = 1
        return activities_occurrences
    
    def calculate_activity_pair_frequencies(self):
        activity_pair_frequencies = {}
        for trace in self.log:
            activities = [event[self.activity_key] for event in trace]
            for i in range(len(activities) - 1):
                current_activity = activities[i]
                next_activity = activities[i+1]
                activity_pair = (current_activity, next_activity)
                activity_pair_frequencies[activity_pair] = activity_pair_frequencies.get(activity_pair, 0) + 1
        return activity_pair_frequencies

    def calculate_weights_for_places(self, activity_pair_frequencies, im):
        weights_places = {}
        for place in self.pn.places:
            if place in im:
                for trace in self.log:
                    weights_places[place] = weights_places.get(place, 0) + 1
            else:
                input_transitions = [arc.source for arc in place.in_arcs]
                output_transitions = [arc.target for arc in place.out_arcs]
                total_activity_pair_frequency = 0
                for input_transition in input_transitions:
                    for output_transition in output_transitions:
                        input_activity = input_transition.label
                        output_activity = output_transition.label
                        total_activity_pair_frequency += activity_pair_frequencies.get((input_activity, output_activity), 0)
                weights_places[place] = max(1,total_activity_pair_frequency)
        return weights_places
    
    def calculate_weights_for_transitions(self, place_weights):
        weight_fork = {}
        activity_frequency = self.scan_log(self.pn)
        for transition in self.pn.transitions:
            input_places = [arc.source for arc in transition.in_arcs]
            for place in input_places:
                if place in place_weights:
                    output_transitions = [arc.target for arc in place.out_arcs]
                    sum_output_transitions = sum(activity_frequency[tran.label] for tran in output_transitions)
                    if transition in weight_fork:
                        weight_fork[transition] += place_weights[place]*activity_frequency[transition.label]/sum_output_transitions
                    else:
                        weight_fork[transition] = place_weights[place]*activity_frequency[transition.label]/sum_output_transitions
        return weight_fork
        
        
class ForkDistributionEstimator:
    def __init__(self, im):
        self.place_weights = {}
        self.activities_weights = {}
        self.im = im

    def estimate_weights_apply(self, log: EventLog, pn: PetriNet):
        frequency_calculator = FrequencyCalculator(log, pn)
        activity_pair_frequencies = frequency_calculator.calculate_activity_pair_frequencies()
        place_weights = frequency_calculator.calculate_weights_for_places(activity_pair_frequencies, self.im)
        spn = StochasticPetriNet(pn)
        self.activities_weights = frequency_calculator.calculate_weights_for_transitions(place_weights)
        return self.estimate_activity_pair_weights(spn)

    # Assign weights to transitions based on event frequencies
    def estimate_activity_pair_weights(self, spn: StochasticPetriNet):
        for transition in spn.transitions:
            weight = self.load_activities_weights(transition)
            transition.weight = weight
        return spn

    # Retrieve the frequency of a specific activity
    def load_activities_weights(self, tran):
        # Use a default value of 0.0 if the activity is not found in the log
        frequency = self.activities_weights[tran]
        return frequency