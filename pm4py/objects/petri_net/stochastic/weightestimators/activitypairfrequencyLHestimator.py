from collections import defaultdict
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from enum import Enum
from pm4py.objects.log.obj import EventLog
from pm4py.util import constants, exec_utils, xes_constants as xes
from typing import Optional, Dict, Any
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.objects.petri_net.obj import PetriNet
import pm4py

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY

class FrequencyCalculator:
    def __init__(self, log: EventLog, parameters: Optional[Dict[Any, Any]] = None):
        """
        Initializes the FrequencyCalculator object.

        Parameters:
        - log: EventLog - Input event log
        - parameters: Optional[Dict[Any, Any]] - Optional parameters for configuration
        """
        if parameters is None:
            parameters = {}
        self.log = log
        self.parameters = parameters

    def calculate_follows_frequency(self, current_activity, dfg, end_activities):
        """
        Calculates the frequency of an activity-pair to a given activity

        Parameters:
        - current_activity (str): The activity for which we want to calculate the frequency.
        - dfg (Dict[Tuple[str, str], int]): A directed flow graph represented as a dictionary of activity-pair and their frequencies.
        - end_activities (Dict[str], int]): A dictionary of end activities and their frequencies.

        Returns:
        - follows_frequency: int - corresponding activity-pair frequency of current_activity.
        """
        follows_frequency = 0
        for (f, t) in dfg:
            if t == current_activity and (t, f) not in dfg:
                if current_activity not in end_activities:
                    follows_frequency += dfg[(f, t)]
        return follows_frequency

    def calculate_start_frequency(self, current_activity, start_activities):
        """
        Calculates the frequency of starting a process with a given activity.

        Parameters:
        - current_activity: str - The activity to analyze
        - start_activities (Dict[str], int]): A dictionary of start activities and their frequencies.

        Returns:
            int: - corresponding activity-pair frequency of current_activity.
        """
        if current_activity in start_activities:
            return start_activities[current_activity]
        return 0

    def calculate_end_frequency(self, current_activity, end_activities):
        """
        Calculates the frequency of ending a process with a given activity.

        Parameters:
        - current_activity: str - The activity to analyze
        - end_activities (Dict[str], int]): A dictionary of end activities and their frequencies.

        Returns:
            int: - corresponding activity-pair frequency of current_activity.
        """
        if current_activity in end_activities:
            return end_activities[current_activity]
        return 0

    def calculate_wlhpair(self):
        """
        Calculates the weights for activities based on left-handed activity pair frequency estimator.

        Returns:
        - activities_weights: Dict[str, float] - Dictionary with activity and its corresponding weighted frequency
        """
        activities_weights = {}
        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, self.parameters, xes.DEFAULT_NAME_KEY)
        activities_occurrences = log_attributes.get_attribute_values(self.log, activity_key, parameters=self.parameters)
        activities = list(activities_occurrences.keys())
        dfg, start_activities, end_activities = pm4py.discover_dfg(self.log)
        for current_activity in activities:
            start_frequency = self.calculate_start_frequency(current_activity, start_activities)
            end_frequency = self.calculate_end_frequency(current_activity, end_activities)
            follows_frequency = self.calculate_follows_frequency(current_activity, dfg, end_activities)
            weight = max(1, start_frequency + end_frequency + follows_frequency)
            activities_weights[current_activity] = weight
        return activities_weights

class ActivityPairLHWeightEstimator:
    def __init__(self):
        """
        Initializes the ActivityPairLHWeightEstimator object.

        Parameters:
        - activities_weights: defaultdict(float) - Dictionary to store weights for activities
        """
        self.activities_weights = defaultdict(float)

    def estimate_weights_apply(self, log: EventLog, pn: PetriNet):
        """
        Estimates transition weights based on left-handed activity pair frequency estimator.

        Parameters:
        - log: EventLog - Input event log
        - pn: PetriNet - Input Petri net

        Returns:
        - spn: StochasticPetriNet - Stochastic Petri net with estimated transition weights
        """
        frequency_calculator = FrequencyCalculator(log)
        self.activities_weights = frequency_calculator.calculate_wlhpair()
        spn = StochasticPetriNet(pn)
        return self.estimate_activity_pair_weights(spn)

    def estimate_activity_pair_weights(self, spn: StochasticPetriNet):
        """
        Assigns weights to transitions in a Stochastic Petri net based on activity pair frequencies.

        Parameters:
        - spn: StochasticPetriNet - Stochastic Petri net

        Returns:
        - spn: StochasticPetriNet - Stochastic Petri net with updated transition weights
        """
        for transition in spn.transitions:
            weight = self.load_activities_weights(transition)
            transition.weight = weight
        return spn

    def load_activities_weights(self, tran):
        """
        Retrieves the frequency of a specific activity.

        Parameters:
        - tran: Transition - Input transition object

        Returns:
        - frequency: float - Frequency of the activity
        """
        activity = tran.label
        # Use a default value of 1.0 if the activity is not found in the log
        frequency = float(self.activities_weights.get(activity, 1))
        return frequency
