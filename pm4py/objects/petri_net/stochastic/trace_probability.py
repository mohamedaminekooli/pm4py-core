from typing import Union

import pandas as pd
import pm4py
from pm4py.objects import petri_net
from pm4py.objects.log.obj import EventLog, EventStream, Trace, Event
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.obj import StochasticTransitionSystem
from pm4py.objects.petri_net.stochastic.utils import convert
from pm4py.objects.transition_system import utils
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from sympy import Eq, solve, symbols


def extract_activities_from_trace(trace):
    """
    Extract activities from a given trace.

    Parameters:
    - trace: Input trace

    Returns:
    - List of activities in the trace
    """
    return [event[DEFAULT_NAME_KEY] for event in trace]

def construct_petri_net_from_trace(trace: Trace):
    """
    Construct a Petri Net from a given trace.

    Parameters:
    - trace: Input trace

    Returns:
    - Tuple containing Petri Net, initial marking, and final marking
    """
    net, initial_marking, final_marking = petri_net.utils.petri_utils.construct_trace_net(trace)
    return net, initial_marking, final_marking

def get_tran_label(tran):
    """
    Get the label of a transition.

    Parameters:
    - tran: Transition name

    Returns:
    - Transition label
    """
    comma_index = tran.find(',')
    closing_parenthesis_index = tran.find(')')
    # Extract the substring between ',' and ')'
    dfa_tran_name = tran[comma_index + 1: closing_parenthesis_index].strip()
    return dfa_tran_name

def construct_product_system(dfa_ts: TransitionSystem, spn_ts: StochasticTransitionSystem) -> StochasticTransitionSystem:
    """
    Construct the product system of a Deterministic Finite Automaton (DFA) and a Stochastic Transition System (SPN).

    Parameters:
    - dfa_ts: Transition system of the DFA
    - spn_ts: Stochastic Transition system of the SPN

    Returns:
    - product_system: StochasticTransitionSystem representing the product system
    """
    product_system = StochasticTransitionSystem()
    dfa_im, spn_im = [], []
    for dfa_state in dfa_ts.states:
        if not dfa_state.incoming or str(dfa_state.incoming) == str({('skip', None)}):
            dfa_im.append(dfa_state)
            
    for spn_state in spn_ts.states:
        if not spn_state.incoming:
            spn_im.append(spn_state)

    for spn_state in spn_ts.states:      
        for dfa_state in dfa_ts.states:
            # Combine DFA state and SPN state to form a new state in the product system
            product_state_name = f"{spn_state.name};{dfa_state.name}"
            product_state = StochasticTransitionSystem.State(name = product_state_name)
            
            if spn_state in spn_im and dfa_state in dfa_im:
                # fix initial state
                product_state.outgoing = spn_state.outgoing
                product_system.states.add(product_state)
                map_next_state=[product_state]
                i=0
                while i< len(map_next_state):
                    product_state = map_next_state[i]
                    for spn_state in spn_ts.states:
                        if spn_state.name == product_state.name.split(';')[0]:
                            product_state.outgoing = spn_state.outgoing
                    for out_transition in product_state.outgoing:
                        product_transition_name = f"{out_transition.name}"
                        product_transition = StochasticTransitionSystem.Transition(
                            name=product_transition_name,
                            from_state= product_state,
                            data=out_transition.data,
                            weight=out_transition.weight
                        )
                        for current_dfa_state in dfa_ts.states:
                            if current_dfa_state.name == product_state.name.split(';')[1]:
                                dfa_state=current_dfa_state

                        for dfa_tran in dfa_state.outgoing:
                            if get_tran_label(str(dfa_tran)) == get_tran_label(product_transition_name):
                                for tran_dfa in dfa_ts.transitions:
                                    if tran_dfa.name==dfa_tran.name and tran_dfa.from_state == dfa_state:
                                        dfa_tran_next = tran_dfa
                                next_state_name = f"{out_transition.to_state};{dfa_tran_next.to_state}"
                                for state in product_system.states:
                                    if state.name == str(next_state_name):
                                        next_state=state
                                        break
                                    else:
                                        next_state = StochasticTransitionSystem.State(name = next_state_name)
                                product_transition.to_state = next_state
                                add_tr=True
                                for tr in product_system.transitions:
                                    if tr.name==product_transition.name and tr.from_state==product_transition.from_state:
                                        add_tr=False
                                        break
                                if add_tr:
                                    product_system.transitions.add(product_transition)
                                next_state.incoming.add(product_transition)
                                if next_state not in product_system.states:
                                    product_system.states.add(next_state)
                                break                       
                        if next_state not in map_next_state:
                            map_next_state.append(next_state)
                    i+=1
    return product_system


def get_system_equations(product_system: StochasticTransitionSystem, 
                        dfa_ts: TransitionSystem, ts: StochasticTransitionSystem):
    """
    Get the system of equations for a product system.

    Parameters:
    - product_system: StochasticTransitionSystem representing the product system
    - dfa_ts: Transition system of the DFA
    - ts: Stochastic Transition system of the SPN

    Returns:
    - List of equations
    """
    final_ts_states_name=[]
    for dfa_state in dfa_ts.states:
        if len(dfa_state.outgoing) == 1:
            final_dfa_state_name = dfa_state.name
    for ts_state in ts.states:
        if len(ts_state.outgoing) == 0:
            final_ts_states_name.append(ts_state.name)
    equations = []
    for state in product_system.states:
        x_state = symbols(f'x_{state.name}')
        if not state.outgoing:
            if final_dfa_state_name == state.name.split(';')[1] and state.name.split(';')[0] in final_ts_states_name:
                equation = Eq(x_state, 1.0)
            else:
                equation = Eq(x_state, 0.0)
        else:
            state_probability=0.0
            to_st=None
            for tran in state.outgoing:
                weight=tran.weight
                for tr in product_system.transitions:
                    if tr.name==tran.name and str(tr.from_state)==state.name:
                        to_st=tr.to_state
                        x = symbols(f'x_{to_st.name}')
                        state_probability+=weight*x
            equation = Eq(x_state, state_probability)
        equations.append(equation)
    return equations

def add_silent_tran_to_reachability_graph(dfa_ts: TransitionSystem) -> TransitionSystem:
    """
    Add silent transitions to the reachability graph of a Deterministic Finite Automaton (DFA).

    Parameters:
    - dfa_ts: Transition system of the DFA

    Returns:
    - Modified DFA transition system with added silent transitions
    """
    for state in dfa_ts.states:
            utils.add_arc_from_to(name=str(("skip", None)), fr=state, to=state, ts=dfa_ts)
    return dfa_ts

def compute_proba_from_trace(spn: StochasticPetriNet, im: Marking, trace: Trace):
    """
    Compute the probability of a trace in a Stochastic Petri Net (SPN).

    Parameters:
    - spn: Stochastic Petri Net (SPN)
    - im: Initial marking
    - trace: Input trace

    Returns:
    - Probability of the trace
    """
    possible_events=[tran.label for tran in spn.transitions]
    for event in trace:
        if event.strip() not in possible_events:
            raise Exception("the method can be applied only to a trace with events corresponding to the existing transition labels of the spn!")
    activity_key = exec_utils.get_param_value(PARAMETER_CONSTANT_ACTIVITY_KEY, parameters=None, default=DEFAULT_NAME_KEY)
    ts = convert.construct_reachability_graph(spn, im)

    case=Trace()
    for act in trace:
        case.append(Event({activity_key: act.strip()}))
    net, im, fm = construct_petri_net_from_trace(case)
    dfa_ts = pm4py.convert_to_reachability_graph(net, im, fm)
    dfa_ts = add_silent_tran_to_reachability_graph(dfa_ts)
    
    product_system = construct_product_system(dfa_ts, ts)
    
    unknowns = symbols([f'x_{state.name}' for state in product_system.states])
    equations = get_system_equations(product_system, dfa_ts, ts)
    # Solve the system of equations
    solution = solve(equations, unknowns, dict=True)
    for variable, value in solution[0].items():
        for product_state in product_system.states:
            if not product_state.incoming:
                if product_state.name in str(variable):
                    probability = value
    return probability

def compute_traces_probability(log: Union[EventLog, pd.DataFrame, EventStream], 
                               spn: StochasticPetriNet, im: Marking):
    """
    Compute the probabilities of traces of a log in a Stochastic Petri Net (SPN).

    Parameters:
    - log: Input event log
    - spn: Stochastic Petri Net
    - im: Initial marking

    Returns:
    - Dictionary containing probabilities for each trace
    """
    if type(log) not in [pd.DataFrame, EventLog, EventStream]:
        raise Exception("the method can be applied only to a traditional event log!")
    if type(log) is not EventLog:
        log=pm4py.convert_to_event_log(log)

    ts = convert.construct_reachability_graph(spn, im)
    
    variants = {}
    for trace in log:
        # Extract activities from the trace
        activities = tuple(extract_activities_from_trace(trace))
        if activities in variants:
            continue
        net, im, fm = construct_petri_net_from_trace(trace)
        dfa_ts = pm4py.convert_to_reachability_graph(net, im, fm)
        dfa_ts = add_silent_tran_to_reachability_graph(dfa_ts)

        product_system = construct_product_system(dfa_ts, ts)

        unknowns = symbols([f'x_{state.name}' for state in product_system.states])
        equations = get_system_equations(product_system, dfa_ts, ts)
        # Solve the system of equations
        solution = solve(equations, unknowns, dict=True)
        
        for variable, value in solution[0].items():
            for product_state in product_system.states:
                if not product_state.incoming:
                    if product_state.name in str(variable):
                        variants[activities] = value
    return variants
