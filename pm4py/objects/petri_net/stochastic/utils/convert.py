import re
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.obj import StochasticTransitionSystem
from pm4py.objects import petri_net
from pm4py.objects.petri_net.utils import align_utils
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
        Stochastic Transition system that represents the Stochastic reachability graph of the input Stochastic Petri net.
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
    net: Stochastic Petri net
    initial_marking: initial marking of the Petri net.

    Returns
    -------
    re_gr: Stochastic Transition system that represents the reachability graph of the input Stochastic Petri net.
    """
    incoming_transitions, outgoing_transitions, eventually_enabled = marking_flow_petri(net, initial_marking,
                                                                                        parameters=parameters)

    return construct_reachability_graph_from_flow(incoming_transitions, outgoing_transitions,
                                                  use_trans_name=use_trans_name, parameters=parameters)