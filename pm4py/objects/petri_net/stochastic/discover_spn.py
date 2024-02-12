from enum import Enum
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
import pandas as pd
from typing import Dict, Any, Union, Optional
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, EventStream

from pm4py.objects.petri_net.stochastic.weightestimators.abstractfrequencyestimator import AbstractFrequencyEstimator
from pm4py.objects.petri_net.stochastic.weightestimators.activitypairfrequencyeRHestimator import ActivityPairRHWeightEstimator
from pm4py.objects.petri_net.stochastic.weightestimators.activitypairfrequencyLHestimator import ActivityPairLHWeightEstimator
from pm4py.objects.petri_net.stochastic.weightestimators.forkdistributionestimator import ForkDistributionEstimator
from pm4py.objects.petri_net.stochastic.weightestimators.meanscaledactivitypairfrequencyestimator import MeanScaledActivityPairFrequencyEstimator
from pm4py.objects.petri_net.stochastic.weightestimators.alignmentestimator import AlignmentEstimator

from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
from pm4py import util as pmutil
from pm4py.util import constants, exec_utils, xes_constants as xes_util
from pm4py.objects.conversion.log import converter as log_converter

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY

def check_log(log, parameters):
    """
    Check the format and structure of the input log.

    Parameters:
    - log: The input log, which can be a Pandas DataFrame, EventLog, or EventStream.
    - parameters: Optional parameters to customize log checking.
    """
    if parameters is None:
        parameters = {}
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, pmutil.constants.CASE_CONCEPT_NAME)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_util.DEFAULT_NAME_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters, xes_util.DEFAULT_TIMESTAMP_KEY)

    if type(log) not in [pd.DataFrame, EventLog, EventStream]:
        raise Exception("the method can be applied only to a traditional event log!")
    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(log, activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key)

def discover_spn_abstract_fraquency_estimator(log: Union[EventLog, pd.DataFrame, EventStream], 
petri_net: PetriNet, parameters: Optional[Dict[Union[str, Parameters], Any]] = None ) -> StochasticPetriNet:
    """
    Discover stochastic weights using AbstractFrequencyEstimator.

    Parameters:
    - log: Input event log.
    - petri_net: Input Petri Net.
    - parameters: Optional parameters.

    Returns:
    - StochasticPetriNet: Stochastic Petri Net with weight estimation.
    """
    check_log(log, parameters)
    discoverer = AbstractFrequencyEstimator()
    return discoverer.estimate_weights_apply(log, petri_net)

def discover_spn_activity_pair_rh_weight_estimator(log: Union[EventLog, pd.DataFrame, EventStream], 
petri_net: PetriNet, parameters: Optional[Dict[Union[str, Parameters], Any]] = None ) -> StochasticPetriNet:
    """
    Discover stochastic weights using ActivityPairRHWeightEstimator.

    Parameters:
    - log: Input event log.
    - petri_net: Input Petri Net.
    - parameters: Optional parameters.

    Returns:
    - StochasticPetriNet: Stochastic Petri Net with weight estimation.
    """
    check_log(log, parameters)
    if type(log) is not pd.DataFrame:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME, parameters=parameters)
    discoverer = ActivityPairRHWeightEstimator()
    return discoverer.estimate_weights_apply(log=log, pn=petri_net)

def discover_spn_activity_pair_lh_weight_estimator(log: Union[EventLog, pd.DataFrame, EventStream], 
petri_net: PetriNet, parameters: Optional[Dict[Union[str, Parameters], Any]] = None ) -> StochasticPetriNet:
    """
    Discover stochastic weights using ActivityPairLHWeightEstimator.

    Parameters:
    - log: Input event log.
    - petri_net: Input Petri Net.
    - parameters: Optional parameters.

    Returns:
    - StochasticPetriNet: Stochastic Petri Net with weight estimation.
    """
    check_log(log, parameters)
    if type(log) is not pd.DataFrame:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME, parameters=parameters)
    discoverer = ActivityPairLHWeightEstimator()
    return discoverer.estimate_weights_apply(log, petri_net)

def discover_spn_fork_distribution_estimator(log: Union[EventLog, pd.DataFrame, EventStream],
petri_net: PetriNet, im: Marking, parameters: Optional[Dict[Union[str, Parameters], Any]] = None ) -> StochasticPetriNet:
    """
    Discover stochastic weights using ForkDistributionEstimator.

    Parameters:
    - log: Input event log.
    - petri_net: Input Petri Net.
    - parameters: Optional parameters.

    Returns:
    - StochasticPetriNet: Stochastic Petri Net with weight estimation.
    """
    check_log(log, parameters)
    if type(log) is not EventLog:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
    discoverer = ForkDistributionEstimator(im)
    return discoverer.estimate_weights_apply(log=log, pn=petri_net)

def discover_spn_mean_scaled_activity_pair_frequency_estimator(log: Union[EventLog, pd.DataFrame, EventStream], 
petri_net: PetriNet, parameters: Optional[Dict[Union[str, Parameters], Any]] = None ) -> StochasticPetriNet:
    """
    Discover stochastic weights using MeanScaledActivityPairFrequencyEstimator.

    Parameters:
    - log: Input event log.
    - petri_net: Input Petri Net.
    - parameters: Optional parameters.

    Returns:
    - StochasticPetriNet: Stochastic Petri Net with weight estimation.
    """
    check_log(log, parameters)
    if type(log) is not pd.DataFrame:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME, parameters=parameters)
    discoverer = MeanScaledActivityPairFrequencyEstimator()
    return discoverer.estimate_weights_apply(log=log, pn=petri_net)

def discover_spn_alignment_estimator(log: Union[EventLog, pd.DataFrame, EventStream], 
petri_net: PetriNet, im: Marking, fm: Marking, parameters: Optional[Dict[Union[str, Parameters], Any]] = None ) -> StochasticPetriNet:
    """
    Discover stochastic weights using AlignmentEstimator.

    Parameters:
    - log: Input event log.
    - petri_net: Input Petri Net.
    - parameters: Optional parameters.

    Returns:
    - StochasticPetriNet: Stochastic Petri Net with weight estimation.
    """
    check_log(log, parameters)
    if type(log) is not EventLog:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
    discoverer = AlignmentEstimator(log, petri_net, im, fm)
    return discoverer.estimate_weights_apply(pn=petri_net)