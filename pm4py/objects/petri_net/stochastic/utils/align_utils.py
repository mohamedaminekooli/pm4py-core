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
from copy import copy

from pm4py.algo.conformance.alignments.petri_net import variants
from pm4py.objects.petri_net.utils import check_soundness
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util.xes_constants import DEFAULT_NAME_KEY, DEFAULT_TRACEID_KEY
from pm4py.objects.log.obj import Trace, Event
import time
from pm4py.util.lp import solver
from pm4py.util import exec_utils
from enum import Enum
import sys
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY, PARAMETER_CONSTANT_CASEID_KEY, CASE_CONCEPT_NAME
import importlib.util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import EventLog, EventStream, Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing, constants
import pandas as pd


class Variants(Enum):
    VERSION_STATE_EQUATION_A_STAR = variants.state_equation_a_star
    VERSION_TWEAKED_STATE_EQUATION_A_STAR = variants.tweaked_state_equation_a_star
    VERSION_DIJKSTRA_NO_HEURISTICS = variants.dijkstra_no_heuristics
    VERSION_DIJKSTRA_LESS_MEMORY = variants.dijkstra_less_memory
    VERSION_DISCOUNTED_A_STAR = variants.discounted_a_star

class Parameters(Enum):
    PARAM_TRACE_COST_FUNCTION = 'trace_cost_function'
    PARAM_MODEL_COST_FUNCTION = 'model_cost_function'
    PARAM_SYNC_COST_FUNCTION = 'sync_cost_function'
    PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE = 'ret_tuple_as_trans_desc'
    PARAM_TRACE_NET_COSTS = "trace_net_costs"
    TRACE_NET_CONSTR_FUNCTION = "trace_net_constr_function"
    TRACE_NET_COST_AWARE_CONSTR_FUNCTION = "trace_net_cost_aware_constr_function"
    PARAM_MAX_ALIGN_TIME_TRACE = "max_align_time_trace"
    PARAM_MAX_ALIGN_TIME = "max_align_time"
    PARAMETER_VARIANT_DELIMITER = "variant_delimiter"
    CASE_ID_KEY = PARAMETER_CONSTANT_CASEID_KEY
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    VARIANTS_IDX = "variants_idx"
    SHOW_PROGRESS_BAR = "show_progress_bar"
    CORES = 'cores'
    BEST_WORST_COST_INTERNAL = "best_worst_cost_internal"
    FITNESS_ROUND_DIGITS = "fitness_round_digits"
    SYNCHRONOUS = "synchronous_dijkstra"
    EXPONENT="theta"
    RETURN_SYNC_COST_FUNCTION = "return_sync_cost_function"


def __variant_mapper(variant):
    if type(variant) is str:
        if variant == "Variants.VERSION_STATE_EQUATION_A_STAR":
            variant = Variants.VERSION_STATE_EQUATION_A_STAR
        elif variant == "Variants.VERSION_TWEAKED_STATE_EQUATION_A_STAR":
            variant = Variants.VERSION_TWEAKED_STATE_EQUATION_A_STAR
        elif variant == "Variants.VERSION_DIJKSTRA_NO_HEURISTICS":
            variant = Variants.VERSION_DIJKSTRA_NO_HEURISTICS
        elif variant == "Variants.VERSION_DIJKSTRA_LESS_MEMORY":
            variant = Variants.VERSION_DIJKSTRA_LESS_MEMORY

    return variant


DEFAULT_VARIANT = Variants.VERSION_DIJKSTRA_LESS_MEMORY
if solver.DEFAULT_LP_SOLVER_VARIANT is not None:
    DEFAULT_VARIANT = __variant_mapper(constants.DEFAULT_ALIGNMENTS_VARIANT)

VERSION_STATE_EQUATION_A_STAR = Variants.VERSION_STATE_EQUATION_A_STAR
VERSION_DIJKSTRA_NO_HEURISTICS = Variants.VERSION_DIJKSTRA_NO_HEURISTICS
VERSION_DIJKSTRA_LESS_MEMORY = Variants.VERSION_DIJKSTRA_LESS_MEMORY

VERSIONS = {Variants.VERSION_DIJKSTRA_NO_HEURISTICS, Variants.VERSION_DIJKSTRA_NO_HEURISTICS,
            Variants.VERSION_DIJKSTRA_LESS_MEMORY}


def apply(obj: Union[EventLog, EventStream, pd.DataFrame, Trace], petri_net: PetriNet, initial_marking: Marking, final_marking: Marking, parameters: Optional[Dict[Any, Any]] = None, variant=DEFAULT_VARIANT) -> Union[typing.AlignmentResult, typing.ListAlignments]:
    if parameters is None:
        parameters = {}
    if isinstance(obj, Trace):
        return apply_trace(obj, petri_net, initial_marking, final_marking, parameters=parameters, variant=variant)
    else:
        return apply_log(obj, petri_net, initial_marking,
                         final_marking, parameters=parameters, variant=variant)


def apply_trace(trace, petri_net, initial_marking, final_marking, parameters=None,
                variant=DEFAULT_VARIANT):
    """
    apply alignments to a trace
    Parameters
    -----------
    trace
        :class:`pm4py.log.log.Trace` trace of events
    petri_net
        :class:`pm4py.objects.petri.petrinet.PetriNet` the model to use for the alignment
    initial_marking
        :class:`pm4py.objects.petri.petrinet.Marking` initial marking of the net
    final_marking
        :class:`pm4py.objects.petri.petrinet.Marking` final marking of the net
    variant
        selected variant of the algorithm, possible values: {\'Variants.VERSION_STATE_EQUATION_A_STAR, Variants.VERSION_DIJKSTRA_NO_HEURISTICS \'}
    parameters
        :class:`dict` parameters of the algorithm, for key \'state_equation_a_star\':
            Parameters.ACTIVITY_KEY -> Attribute in the log that contains the activity
            Parameters.PARAM_MODEL_COST_FUNCTION ->
            mapping of each transition in the model to corresponding synchronous costs
            Parameters.PARAM_SYNC_COST_FUNCTION ->
            mapping of each transition in the model to corresponding model cost
            Parameters.PARAM_TRACE_COST_FUNCTION ->
            mapping of each index of the trace to a positive cost value
    Returns
    -----------
    alignment
        :class:`dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and
        **traversed_arcs**
        The alignment is a sequence of labels of the form (a,t), (a,>>), or (>>,t)
        representing synchronous/log/model-moves.
    """
    if parameters is None:
        parameters = copy({PARAMETER_CONSTANT_ACTIVITY_KEY: DEFAULT_NAME_KEY})

    variant = __variant_mapper(variant)
    parameters = copy(parameters)
    best_worst_cost = exec_utils.get_param_value(Parameters.BEST_WORST_COST_INTERNAL, parameters,
                                                 __get_best_worst_cost(petri_net, initial_marking, final_marking, variant, parameters))

    ali = apply_search(trace, petri_net, initial_marking, final_marking,
                                                 parameters=parameters)

    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, [])
    # Instead of using the length of the trace, use the sum of the trace cost function
    trace_cost_function_sum = sum(trace_cost_function)

    if ali is not None and best_worst_cost is not None:
        ltrace_bwc = trace_cost_function_sum + best_worst_cost

        fitness_num = ali['cost'] // STD_MODEL_LOG_MOVE_COST
        fitness_den = ltrace_bwc // STD_MODEL_LOG_MOVE_COST
        fitness = 1 - fitness_num / fitness_den if fitness_den > 0 else 0

        # other possibility: avoid integer division but proceed to rounding.
        # could lead to small differences with respect to the adopted-since-now fitness
        # (since it is rounded)

        """
        initial_trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
        initial_model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
        initial_sync_cost_function = exec_utils.get_param_value(Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None)
        uses_standard_cost_function = initial_trace_cost_function is None and initial_model_cost_function is None and \
                                    initial_sync_cost_function is None
            
        fitness = 1 - ali['cost'] / ltrace_bwc if ltrace_bwc > 0 else 0
        fitness_round_digits = exec_utils.get_param_value(Parameters.FITNESS_ROUND_DIGITS, parameters, 3)
        fitness = round(fitness, fitness_round_digits)
        """

        ali["fitness"] = fitness
        # returning also the best worst cost, for log fitness computation
        ali["bwc"] = ltrace_bwc

    return ali


def apply_log(log, petri_net, initial_marking, final_marking, parameters=None, variant=DEFAULT_VARIANT):
    """
    apply alignments to a log
    Parameters
    -----------
    log
        object of the form :class:`pm4py.log.log.EventLog` event log
    petri_net
        :class:`pm4py.objects.petri.petrinet.PetriNet` the model to use for the alignment
    initial_marking
        :class:`pm4py.objects.petri.petrinet.Marking` initial marking of the net
    final_marking
        :class:`pm4py.objects.petri.petrinet.Marking` final marking of the net
    variant
        selected variant of the algorithm, possible values: {\'Variants.VERSION_STATE_EQUATION_A_STAR, Variants.VERSION_DIJKSTRA_NO_HEURISTICS \'}
    parameters
        :class:`dict` parameters of the algorithm,

    Returns
    -----------
    alignment
        :class:`list` of :class:`dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and
        **traversed_arcs**
        The alignment is a sequence of labels of the form (a,t), (a,>>), or (>>,t)
        representing synchronous/log/model-moves.
    """
    if parameters is None:
        parameters = dict()

    if solver.DEFAULT_LP_SOLVER_VARIANT is not None:
        if not check_soundness.check_easy_soundness_net_in_fin_marking(petri_net, initial_marking, final_marking):
            raise Exception("trying to apply alignments on a Petri net that is not a easy sound net!!")

    variant = __variant_mapper(variant)

    start_time = time.time()
    max_align_time = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME, parameters,
                                                sys.maxsize)
    max_align_time_case = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                     sys.maxsize)

    best_worst_cost = __get_best_worst_cost(petri_net, initial_marking, final_marking, variant, parameters)
    variants_idxs, one_tr_per_var = __get_variants_structure(log, parameters)
    progress = __get_progress_bar(len(one_tr_per_var), parameters)
    parameters[Parameters.BEST_WORST_COST_INTERNAL] = best_worst_cost

    all_alignments = []
    for trace in one_tr_per_var:
        this_max_align_time = min(max_align_time_case, (max_align_time - (time.time() - start_time)) * 0.5)
        parameters[Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = this_max_align_time
        all_alignments.append(apply_trace(trace, petri_net, initial_marking, final_marking, parameters=copy(parameters),
                                          variant=variant))
        if progress is not None:
            progress.update()

    alignments = __form_alignments(variants_idxs, all_alignments)
    __close_progress_bar(progress)

    return alignments


def __get_best_worst_cost(petri_net, initial_marking, final_marking, variant, parameters):
    parameters_best_worst = copy(parameters)

    best_worst_cost = exec_utils.get_variant(variant).get_best_worst_cost(petri_net, initial_marking, final_marking,
                                                                          parameters=parameters_best_worst)

    return best_worst_cost


def __get_variants_structure(log, parameters):
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)

    variants_idxs = {}
    one_tr_per_var = []

    if type(log) is pd.DataFrame:
        case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, CASE_CONCEPT_NAME)
        traces = list(log.groupby(case_id_key)[activity_key].apply(tuple))
        for idx, trace in enumerate(traces):
            if trace not in variants_idxs:
                variants_idxs[trace] = [idx]
                case = Trace()
                for act in trace:
                    case.append(Event({activity_key: act}))
                one_tr_per_var.append(case)
            else:
                variants_idxs[trace].append(idx)
    else:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
        for idx, case in enumerate(log):
            trace = tuple(x[activity_key] for x in case)
            if trace not in variants_idxs:
                variants_idxs[trace] = [idx]
                one_tr_per_var.append(case)
            else:
                variants_idxs[trace].append(idx)

    return variants_idxs, one_tr_per_var


def __get_progress_bar(num_variants, parameters):
    show_progress_bar = exec_utils.get_param_value(Parameters.SHOW_PROGRESS_BAR, parameters, constants.SHOW_PROGRESS_BAR)
    progress = None
    if importlib.util.find_spec("tqdm") and show_progress_bar and num_variants > 1:
        from tqdm.auto import tqdm
        progress = tqdm(total=num_variants, desc="aligning log, completed variants :: ")
    return progress


def __form_alignments(variants_idxs, all_alignments):
    al_idx = {}
    for index_variant, variant in enumerate(variants_idxs):
        for trace_idx in variants_idxs[variant]:
            al_idx[trace_idx] = all_alignments[index_variant]

    alignments = []
    for i in range(len(al_idx)):
        alignments.append(al_idx[i])

    return alignments


def __close_progress_bar(progress):
    if progress is not None:
        progress.close()
    del progress


#--------------------------------

import heapq
import sys
import time
from copy import copy
from enum import Enum

import numpy as np

from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.lp import solver as lp_solver
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing

PARAM_TRACE_COST_FUNCTION = Parameters.PARAM_TRACE_COST_FUNCTION.value
PARAM_MODEL_COST_FUNCTION = Parameters.PARAM_MODEL_COST_FUNCTION.value
PARAM_SYNC_COST_FUNCTION = Parameters.PARAM_SYNC_COST_FUNCTION.value


def apply_search(trace: Trace, petri_net: PetriNet, initial_marking: Marking, final_marking: Marking, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> typing.AlignmentResult:
    """
    Performs the basic alignment search, given a trace and a net.

    Parameters
    ----------
    trace: :class:`list` input trace, assumed to be a list of events (i.e. the code will use the activity key
    to get the attributes)
    petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
    parameters: :class:`dict` (optional) dictionary containing one of the following:
        Parameters.PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
        Parameters.PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        model cost
        Parameters.PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
        synchronous costs
        Parameters.ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events

    Returns
    -------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)
    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    trace_net_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_CONSTR_FUNCTION, parameters,
                                                           None)
    trace_net_cost_aware_constr_function = exec_utils.get_param_value(Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
                                                                      parameters, construct_trace_net_cost_aware)

    if trace_cost_function is None:
        trace_cost_function = list(
            map(lambda e: STD_MODEL_LOG_MOVE_COST, trace))
        parameters[Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

    if model_cost_function is None:
        # reset variables value
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in petri_net.transitions:
            if t.label is not None:
                model_cost_function[t] = STD_MODEL_LOG_MOVE_COST
                sync_cost_function[t] = STD_SYNC_COST
            else:
                model_cost_function[t] = STD_TAU_COST
        parameters[Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
        parameters[Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function

    if trace_net_constr_function is not None:
        # keep the possibility to pass TRACE_NET_CONSTR_FUNCTION in this old version
        trace_net, trace_im, trace_fm = trace_net_constr_function(trace, activity_key=activity_key)
    else:
        trace_net, trace_im, trace_fm, parameters[
            Parameters.PARAM_TRACE_NET_COSTS] = trace_net_cost_aware_constr_function(trace,
                                                                                     trace_cost_function,
                                                                                     activity_key=activity_key)

    alignment = apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, parameters)

    return alignment


def apply_from_variant(variant, petri_net, initial_marking, final_marking, parameters=None):
    """
    Apply the alignments from the specification of a single variant

    Parameters
    -------------
    variant
        Variant (as string delimited by the "variant_delimiter" parameter)
    petri_net
        Petri net
    initial_marking
        Initial marking
    final_marking
        Final marking
    parameters
        Parameters of the algorithm (same as 'apply_search' method, plus 'variant_delimiter' that is , by default)

    Returns
    ------------
    dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
    """
    if parameters is None:
        parameters = {}
    trace = variants_util.variant_to_trace(variant, parameters=parameters)

    return apply_search(trace, petri_net, initial_marking, final_marking, parameters=parameters)

def apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, parameters=None):
    """
        Performs the basic alignment search, given a trace net and a net.

        Parameters
        ----------
        trace: :class:`list` input trace, assumed to be a list of events (i.e. the code will use the activity key
        to get the attributes)
        petri_net: :class:`pm4py.objects.petri.net.PetriNet` the Petri net to use in the alignment
        initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the Petri net
        final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the Petri net
        parameters: :class:`dict` (optional) dictionary containing one of the following:
            Parameters.PARAM_TRACE_COST_FUNCTION: :class:`list` (parameter) mapping of each index of the trace to a positive cost value
            Parameters.PARAM_MODEL_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
            model cost
            Parameters.PARAM_SYNC_COST_FUNCTION: :class:`dict` (parameter) mapping of each transition in the model to corresponding
            synchronous costs
            Parameters.ACTIVITY_KEY: :class:`str` (parameter) key to use to identify the activity described by the events
            Parameters.PARAM_TRACE_NET_COSTS: :class:`dict` (parameter) mapping between transitions and costs

        Returns
        -------
        dictionary: `dict` with keys **alignment**, **cost**, **visited_states**, **queued_states** and **traversed_arcs**
        """
    if parameters is None:
        parameters = {}

    ret_tuple_as_trans_desc = exec_utils.get_param_value(Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE,
                                                         parameters, False)

    trace_cost_function = exec_utils.get_param_value(Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    sync_cost_function = exec_utils.get_param_value(Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None)
    trace_net_costs = exec_utils.get_param_value(Parameters.PARAM_TRACE_NET_COSTS, parameters, None)

    if trace_cost_function is None or model_cost_function is None or sync_cost_function is None:
        sync_prod, sync_initial_marking, sync_final_marking = construct(trace_net, trace_im,
                                                                        trace_fm, petri_net,
                                                                        initial_marking,
                                                                        final_marking,
                                                                        SKIP)
        cost_function = construct_standard_cost_function(sync_prod, SKIP)
    else:
        revised_sync = dict()
        for t_trace in trace_net.transitions:
            for t_model in petri_net.transitions:
                if t_trace.label == t_model.label:
                    revised_sync[(t_trace, t_model)] = sync_cost_function[t_model]

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = construct_cost_aware(
            trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, SKIP,
            trace_net_costs, model_cost_function, revised_sync)

    max_align_time_trace = exec_utils.get_param_value(Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                      sys.maxsize)

    alignment = apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function,
                           SKIP, ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                           max_align_time_trace=max_align_time_trace)

    return_sync_cost = exec_utils.get_param_value(Parameters.RETURN_SYNC_COST_FUNCTION, parameters, False)
    if return_sync_cost:
        # needed for the decomposed alignments (switching them from state_equation_less_memory)
        return alignment, cost_function

    return alignment


def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, skip, ret_tuple_as_trans_desc=False,
                    max_align_time_trace=sys.maxsize):
    """
    Performs the basic alignment search on top of the synchronous product net, given a cost function and skip-symbol

    Parameters
    ----------
    sync_prod: :class:`pm4py.objects.petri.net.PetriNet` synchronous product net
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the synchronous product net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the synchronous product net
    cost_function: :class:`dict` cost function mapping transitions to the synchronous product net
    skip: :class:`Any` symbol to use for skips in the alignment

    Returns
    -------
    dictionary : :class:`dict` with keys **alignment**, **cost**, **visited_states**, **queued_states**
    and **traversed_arcs**
    """
    return __search(sync_prod, initial_marking, final_marking, cost_function, skip,
                    ret_tuple_as_trans_desc=ret_tuple_as_trans_desc, max_align_time_trace=max_align_time_trace)


def __search(sync_net, ini, fin, cost_function, skip, ret_tuple_as_trans_desc=False,
             max_align_time_trace=sys.maxsize):
    start_time = time.time()

    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    incidence_matrix = inc_mat_construct(sync_net)
    ini_vec, fin_vec, cost_vec = __vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)

    closed = set()

    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
    cost_vec = [x * 1.0 for x in cost_vec]

    use_cvxopt = False
    if lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN or lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP:
        use_cvxopt = True

    if use_cvxopt:
        # not available in the latest version of PM4Py
        from cvxopt import matrix

        a_matrix = matrix(a_matrix)
        g_matrix = matrix(g_matrix)
        h_cvx = matrix(h_cvx)
        cost_vec = matrix(cost_vec)

    h, x = __compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                       ini,
                                                       fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                       use_cvxopt=use_cvxopt)
    ini_state = SearchTuple(0 + h, 0, h, ini, None, None, x, True)
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1

    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    while not len(open_set) == 0:
        if (time.time() - start_time) > max_align_time_trace:
            return None

        curr = heapq.heappop(open_set)

        current_marking = curr.m

        while not curr.trust:
            if (time.time() - start_time) > max_align_time_trace:
                return None

            already_closed = current_marking in closed
            if already_closed:
                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            h, x = __compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                               incidence_matrix, curr.m,
                                                               fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                               use_cvxopt=use_cvxopt)
            lp_solved += 1

            # 11/10/19: shall not a state for which we compute the exact heuristics be
            # by nature a trusted solution?
            tp = SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
            # 11/10/2019 (optimization ZA) heappushpop is slightly more efficient than pushing
            # and popping separately
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = current_marking in closed
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            if current_marking == fin:
                return __reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=lp_solved)

        closed.add(current_marking)
        visited += 1

        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and __is_log_move(t, skip) and __is_model_move(t, skip))]

        for t, cost in trans_to_visit_with_cost:
            traversed += 1
            new_marking = add_markings(current_marking, t.add_marking)

            if new_marking in closed:
                continue
            g = curr.g + cost

            queued += 1
            h, x = __derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            trustable = __trust_solution(x)
            new_f = g + h

            tp = SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
            heapq.heappush(open_set, tp)
#-----------------
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
import heapq
import sys
from copy import copy
from typing import List, Tuple

import numpy as np

from pm4py.objects.petri_net import semantics, properties
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.util.lp import solver as lp_solver

SKIP = '>>'
STD_MODEL_LOG_MOVE_COST = 10000
STD_TAU_COST = 1
STD_SYNC_COST = 0



def construct_standard_cost_function(synchronous_product_net, skip):
    """
    Returns the standard cost function, which is:
    * event moves: cost 1000
    * model moves: cost 1000
    * tau moves: cost 1
    * sync moves: cost 0
    :param synchronous_product_net:
    :param skip:
    :return:
    """
    costs = {}
    for t in synchronous_product_net.transitions:
        if (skip == t.label[0] or skip == t.label[1]) and (t.label[0] is not None and t.label[1] is not None):
            costs[t] = STD_MODEL_LOG_MOVE_COST
        else:
            if skip == t.label[0] and t.label[1] is None:
                costs[t] = STD_TAU_COST
            else:
                costs[t] = STD_SYNC_COST
    return costs


def add_markings(curr, add):
    m = Marking()
    for p in curr.items():
        m[p[0]] = p[1]
    for p in add.items():
        m[p[0]] += p[1]
        if m[p[0]] == 0:
            del m[p[0]]
    return m

def __reconstruct_alignment(state, visited, queued, traversed, ret_tuple_as_trans_desc=False, lp_solved=0):
    alignment = list()
    silent_occurence = {}
    if state.p is not None and state.t is not None:
        parent = state.p
        if ret_tuple_as_trans_desc:
            alignment = [(state.t.name, state.t.label)]
            while parent.p is not None:
                alignment = [(parent.t.name, parent.t.label)] + alignment
                parent = parent.p
        else:
            alignment = [state.t.label]
            if state.t.label[1] == None:
                    silent_occurence[state.t.name[1]] = silent_occurence.get(state.t.name[1],0) + 1
            while parent.p is not None:
                alignment = [parent.t.label] + alignment
                if parent.t.label[1] == None:
                    silent_occurence[parent.t.name[1]] = silent_occurence.get(parent.t.name[1],0) + 1
                parent = parent.p
    return {'alignment': alignment, 'cost': state.g, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed, 'lp_solved': lp_solved, 'silent_occurence': silent_occurence}

def __derive_heuristic(incidence_matrix, cost_vec, x, t, h):
    x_prime = x.copy()
    x_prime[incidence_matrix.transitions[t]] -= 1
    return max(0, h - cost_vec[incidence_matrix.transitions[t]]), x_prime


def __is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def __is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip


def __trust_solution(x):
    for v in x:
        if v < -0.001:
            return False
    return True


def __compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                          marking, fin_vec, variant, use_cvxopt=False, strict=True):
    m_vec = incidence_matrix.encode_marking(marking)
    b_term = [i - j for i, j in zip(fin_vec, m_vec)]
    b_term = np.matrix([x * 1.0 for x in b_term]).transpose()

    if not strict:
        g_matrix = np.vstack([g_matrix, a_matrix])
        h_cvx = np.vstack([h_cvx, b_term])
        a_matrix = np.zeros((0, a_matrix.shape[1]))
        b_term = np.zeros((0, b_term.shape[1]))

    if use_cvxopt:
        # not available in the latest version of PM4Py
        from cvxopt import matrix

        b_term = matrix(b_term)

    parameters_solving = {"solver": "glpk"}

    sol = lp_solver.apply(cost_vec, g_matrix, h_cvx, a_matrix, b_term, parameters=parameters_solving,
                          variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)

    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = points if points is not None else [0.0] * len(sync_net.transitions)

    return prim_obj, points

def __vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function):
    ini_vec = incidence_matrix.encode_marking(ini)
    fini_vec = incidence_matrix.encode_marking(fin)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
    return ini_vec, fini_vec, cost_vec


class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
        elif self.trust and not other.trust:
            return True
        else:
            return self.h < other.h

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)