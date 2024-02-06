from collections import defaultdict

import pandas as pd
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.util import constants, exec_utils, xes_constants as xes
from typing import Optional, Dict, Any, Union
from pm4py.objects.petri_net.obj import PetriNet
from enum import Enum

# Enum class for defining parameters
class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY

# Class for estimating transition weights based on activity frequencies
class AbstractFrequencyEstimator:
    def __init__(self):
        """
        Initializes the AbstractFrequencyEstimator object.

        Attributes:
        - activity_frequency: defaultdict(float) - Dictionary to store activity frequencies
        """
        self.activity_frequency = defaultdict(int)

    def estimate_weights_apply(self, log: EventLog, pn: PetriNet, parameters: Optional[Dict[Any, Any]] = None):
        """
        Estimates transition weights based on activity frequencies.

        Parameters:
        - log: EventLog - Input event log
        - pn: PetriNet - Input Petri net
        - parameters: Optional[Dict[Any, Any]] - Optional parameters for configuration

        Returns:
        - spn: StochasticPetriNet - Stochastic Petri net with estimated transition weights
        """
        self.activity_frequency = self.scan_log(log, pn, parameters)
        spn = StochasticPetriNet(pn)
        return self.estimate_weights_activity_frequencies(spn)

    def scan_log(self, log: Union[EventLog, pd.DataFrame, EventStream], pn: PetriNet, parameters: Optional[Dict[Any, Any]] = None):
        """
        Scans the event log and updates the activity frequencies.

        Parameters:
        - log: EventLog - Input event log
        - pn: PetriNet - Input Petri net
        - parameters: Optional[Dict[Any, Any]] - Optional parameters for configuration

        Returns:
        - activities_occurrences: Dict[str, float] - Dictionary with activity and its corresponding frequency
        """
        if parameters is None:
            parameters = {}

        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes.DEFAULT_NAME_KEY)
        activities_occurrences = log_attributes.get_attribute_values(log, activity_key, parameters=parameters)
        for transition in pn.transitions:
            if transition.label not in activities_occurrences:
                activities_occurrences[transition.label] = 1
        return activities_occurrences

    def estimate_weights_activity_frequencies(self, spn: StochasticPetriNet):
        """
        Assigns weights to transitions in a Stochastic Petri net based on activity frequencies.

        Parameters:
        - spn: StochasticPetriNet - Stochastic Petri net with transitions

        Returns:
        - spn: StochasticPetriNet - Stochastic Petri net with updated transition weights
        """
        for transition in spn.transitions:
            weight = self.load_activity_frequency(transition)
            transition.weight = weight
        return spn

    def load_activity_frequency(self, tran):
        """
        Retrieves the frequency of a specific activity.

        Parameters:
        - tran: Transition - Input transition object

        Returns:
        - frequency: float - Frequency of the activity
        """
        activity = tran.label
        # Use a default value of 1.0 if the activity is not found in the log
        frequency = float(self.activity_frequency.get(activity, 1.0))
        return frequency
