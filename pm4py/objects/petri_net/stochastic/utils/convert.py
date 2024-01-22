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
import re
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.stochastic_transition_system import StochasticTransitionSystem
from pm4py.objects import petri_net
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.objects.petri_net.utils import align_utils
from pm4py.objects.transition_system import obj as ts
from pm4py.objects.transition_system import utils
from pm4py.util import exec_utils
from enum import Enum
import time


class Parameters(Enum):
    MAX_ELAB_TIME = "max_elab_time"
    PETRI_SEMANTICS = "petri_semantics"


def staterep(name):
    """
    Creates a string representation for a state of a transition system.
    Necessary because graphviz does not support symbols simulation than alphanimerics and '_'.
    TODO: find a better representation.

    Parameters
    ----------
    name: the name of a state

    Returns
    -------
    Version of the name filtered of non-alphanumerical characters (except '_').
    """
    return re.sub(r'\W+', '', name)


def marking_flow_petri(net: StochasticPetriNet, im, return_eventually_enabled=False, parameters=None):
    """
    Construct the marking flow of a Stochastic Petri net

    Parameters
    -----------------
    net
        Stochastic Petri net
    im
        Initial marking
    return_eventually_enabled
        Return the eventually enabled (visible) transitions
    """

    if parameters is None:
        parameters = {}

    # set a maximum execution time of 1 day (it can be changed by providing the parameter)
    max_exec_time = exec_utils.get_param_value(Parameters.MAX_ELAB_TIME, parameters, 86400)
    semantics = exec_utils.get_param_value(Parameters.PETRI_SEMANTICS, parameters, petri_net.semantics.ClassicSemantics())

    start_time = time.time()

    incoming_transitions = {im: set()}
    outgoing_transitions = {}
    eventually_enabled = {}
    active = [im]
    while active:
        if (time.time() - start_time) >= max_exec_time:
            # interrupt the execution
            return incoming_transitions, outgoing_transitions, eventually_enabled
        m = active.pop()
        enabled_transitions = semantics.enabled_transitions(net, m)
        if return_eventually_enabled:
            eventually_enabled[m] = align_utils.get_visible_transitions_eventually_enabled_by_marking(net, m)
        outgoing_transitions[m] = {}
        sum_weight = 0.0
        for t in enabled_transitions:
            sum_weight += t.weight
        transition_probability = {}
        for t in enabled_transitions:
            transition_probability[t] = t.weight / sum_weight
            tran_proba = (t, transition_probability[t])
            nm = semantics.weak_execute(t, net, m)
            outgoing_transitions[m][tran_proba] = nm
            if nm not in incoming_transitions:
                incoming_transitions[nm] = set()
                if nm not in active:
                    active.append(nm)
            incoming_transitions[nm].add(tran_proba)
    # print(f"incoming: {incoming_transitions}")
    # print(f"outgoing: {outgoing_transitions}")

    return incoming_transitions, outgoing_transitions, eventually_enabled


def construct_reachability_graph_from_flow(incoming_transitions, outgoing_transitions,
                                           use_trans_name=False, parameters=None):
    """
    Construct the reachability graph from the marking flow

    Parameters
    ----------------
    incoming_transitions
        Incoming transitions
    outgoing_transitions
        Outgoing transitions
    use_trans_name
        Use the transition name

    Returns
    ----------------
    re_gr
        Transition system that represents the reachability graph of the input Petri net.
    """
    if parameters is None:
        parameters = {}

    re_gr = StochasticTransitionSystem()

    map_states = {}
    for s in incoming_transitions:
        map_states[s] = StochasticTransitionSystem.State(staterep(repr(s)))
        re_gr.states.add(map_states[s])

    for s1 in outgoing_transitions:
        for tran_proba in outgoing_transitions[s1]:
            s2 = outgoing_transitions[s1][tran_proba]
            if use_trans_name:
                add_arc_from_to(tran_proba[0].name, tran_proba[1], map_states[s1], map_states[s2], re_gr)
            else:
                add_arc_from_to(repr(tran_proba[0]), tran_proba[1], map_states[s1], map_states[s2], re_gr)

    return re_gr


def add_arc_from_to(name, tran_proba, fr, to, ts, data=None):
    """
    Adds a transition from a state to another state in some transition system.
    Assumes from and to are in the transition system!

    Parameters
    ----------
    name: name of the transition
    fr: state from
    to:  state to
    ts: transition system to use
    data: data associated to the Transition System

    Returns
    -------
    None
    """
    tran = StochasticTransitionSystem.Transition(name, fr, to, data, tran_proba)
    ts.transitions.add(tran)
    fr.outgoing.add(tran)
    to.incoming.add(tran)

def construct_reachability_graph(net: StochasticPetriNet, initial_marking, use_trans_name=False, parameters=None) -> StochasticTransitionSystem:
    """
    Creates a reachability graph of a certain Stochastic Petri net.
    DO NOT ATTEMPT WITH AN UNBOUNDED PETRI NET, EVER.

    Parameters
    ----------
    net: Petri net
    initial_marking: initial marking of the Petri net.

    Returns
    -------
    re_gr: Transition system that represents the reachability graph of the input Petri net.
    """
    incoming_transitions, outgoing_transitions, eventually_enabled = marking_flow_petri(net, initial_marking,
                                                                                        parameters=parameters)

    return construct_reachability_graph_from_flow(incoming_transitions, outgoing_transitions,
                                                  use_trans_name=use_trans_name, parameters=parameters)

#--------------------------------

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
__doc__ = """
The ``pm4py.convert`` module contains the cross-conversions implemented in ``pm4py``
"""

from typing import Union, Tuple, Optional, Collection, List, Any

import pandas as pd
from copy import deepcopy

from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.powl.obj import POWL
from pm4py.objects.heuristics_net.obj import HeuristicsNet
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.util import constants
from pm4py.utils import get_properties, __event_log_deprecation_warning
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
import networkx as nx


def convert_to_petri_net(*args: Union[BPMN, ProcessTree, HeuristicsNet, POWL, dict]) -> Tuple[PetriNet, Marking, Marking]:
    """
    Converts an input model to an (accepting) Petri net.
    The input objects can either be a process tree, BPMN model or a Heuristic net.
    The output is a triple, containing the Petri net and the initial and final markings. The markings are only returned if they can be reasonable derived from the input model.

    :param args: process tree, Heuristics net, BPMN or POWL model
    :rtype: ``Tuple[PetriNet, Marking, Marking]``
    
    .. code-block:: python3

       import pm4py

       # imports a process tree from a PTML file
       process_tree = pm4py.read_ptml("tests/input_data/running-example.ptml")
       net, im, fm = pm4py.convert_to_petri_net(process_tree)
    """
    if isinstance(args[0], PetriNet):
        # the object is already a Petri net
        return args[0], args[1], args[2]
    elif isinstance(args[0], ProcessTree):
        if isinstance(args[0], POWL):
            from pm4py.objects.conversion.powl import converter
            return converter.apply(args[0])
        from pm4py.objects.conversion.process_tree.variants import to_petri_net
        return to_petri_net.apply(args[0])
    elif isinstance(args[0], BPMN):
        from pm4py.objects.conversion.bpmn.variants import to_petri_net
        return to_petri_net.apply(args[0])
    elif isinstance(args[0], HeuristicsNet):
        from pm4py.objects.conversion.heuristics_net.variants import to_petri_net
        return to_petri_net.apply(args[0])
    elif isinstance(args[0], dict):
        # DFG
        from pm4py.objects.conversion.dfg.variants import to_petri_net_activity_defines_place
        return to_petri_net_activity_defines_place.apply(args[0], parameters={
            to_petri_net_activity_defines_place.Parameters.START_ACTIVITIES: args[1],
            to_petri_net_activity_defines_place.Parameters.END_ACTIVITIES: args[2]})
    # if no conversion is done, then the format of the arguments is unsupported
    raise Exception("unsupported conversion of the provided object to Petri net")


def convert_to_reachability_graph(*args: Union[Tuple[PetriNet, Marking, Marking], BPMN, ProcessTree]) -> StochasticTransitionSystem:
    """
    Converts an input model to a reachability graph (transition system).
    The input models can either be Petri nets (with markings), BPMN models or process trees.
    The output is the state-space of the model (i.e., the reachability graph), enocdoed as a ``StochasticTransitionSystem`` object.

    :param args: petri net (along with initial and final marking), process tree or BPMN
    :rtype: ``StochasticTransitionSystem``
    
    .. code-block:: python3

        import pm4py

        # reads a Petri net from a file
        net, im, fm = pm4py.read_pnml("tests/input_data/running-example.pnml")
        # converts it to reachability graph
        reach_graph = pm4py.convert_to_reachability_graph(net, im, fm)
    """
    if isinstance(args[0], PetriNet):
        net, im, fm = args[0], args[1], args[2]
    else:
        net, im, fm = convert_to_petri_net(*args)

    from pm4py.objects.petri_net.utils import reachability_graph
    return reachability_graph.construct_reachability_graph(net, im)