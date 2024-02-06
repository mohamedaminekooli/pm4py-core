import os
import sys
sys.path.insert(0, 'C:\\Users\\MohamedAmineKooli\\BA\\pm4py-core')
from pm4py.objects.petri_net.stochastic.utils import convert
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.objects import petri_net
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.stochastic_transition_system import StochasticTransitionSystem
from sympy import symbols, Eq, solve
import pm4py
from pm4py.objects.transition_system import utils
from pm4py.util import exec_utils

def extract_activities_from_trace(trace):
    """
    Extract activities from a given trace.

    Parameters:
    - trace: Input trace

    Returns:
    - List of activities in the trace
    """
    return [event[DEFAULT_NAME_KEY] for event in trace]

def construct_petri_net_from_trace(trace):
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

def construct_product_system(dfa_ts, spn_ts):
    """
    Construct the product system of a Deterministic Finite Automaton (DFA) and a Stochastic Transition System (SPN).

    Parameters:
    - dfa_ts: Transition system of the DFA
    - spn_ts: Transition system of the SPN

    Returns:
    - product_system: StochasticTransitionSystem representing the product system
    """
    product_system = StochasticTransitionSystem()
    dfa_im, dfa_fm, spn_im, spn_fm = [], [], [], []
    for dfa_state in dfa_ts.states:
        if not dfa_state.incoming or str(dfa_state.incoming) == str({('skip', None)}):
            dfa_im.append(dfa_state)
        if not dfa_state.outgoing or str(dfa_state.outgoing) == str({('skip', None)}):
            dfa_fm.append(dfa_state)
            
    for spn_state in spn_ts.states:
        if not spn_state.incoming:
            spn_im.append(spn_state)
        if not spn_state.outgoing:
            spn_fm.append(spn_state)

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
                        # exist=False
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


from sympy import symbols, Eq, solve
def get_system_equations(product_system, dfa_ts, ts):
    """
    Get the system of equations for a product system.

    Parameters:
    - product_system: StochasticTransitionSystem representing the product system
    - dfa_ts: Transition system of the DFA
    - ts: Transition system of the SPN

    Returns:
    - List of equations
    """
    final_ts_states_name=[]
    for tran in product_system.transitions:
        print(f"{tran}: {tran.weight}")
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
                print(f"{x_state} is final is 1.0")
            else:
                equation = Eq(x_state, 0.0)
                print(f"{x_state} is final is 0.0")
        else:
            val=0.0
            to_st=None
            for tran in state.outgoing:
                weight=tran.weight
                # print(state)
                # print(tran)
                for tr in product_system.transitions:
                    if tr.name==tran.name and str(tr.from_state)==state.name:
                        to_st=tr.to_state
                #print(f"{tran}={weight}")
                        x = symbols(f'x_{to_st.name}')
                        val+=weight*x
            #print(f"{x_state}={val}")
            print(f"{x_state} is {val}")
            equation = Eq(x_state, val)
        equations.append(equation)
    return equations

def add_silent_tran_to_reachability_graph(dfa_ts):
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

def compute_proba_from_trace(spn, im, trace):
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
    # show ts
    from pm4py.objects.petri_net.stochastic.stochastic_transition_system import visualizer as ts_visualizer
    gviz = ts_visualizer.apply(ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
    ts_visualizer.view(gviz)
    case=Trace()
    for act in trace:
        case.append(Event({activity_key: act.strip()}))
    net, im, fm = construct_petri_net_from_trace(case)
    dfa_ts = pm4py.convert_to_reachability_graph(net, im, fm)
    dfa_ts = add_silent_tran_to_reachability_graph(dfa_ts)
    # show dfa_ts
    from pm4py.visualization.transition_system import visualizer as ts_visualizer
    gviz = ts_visualizer.apply(dfa_ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
    ts_visualizer.view(gviz)
    product_system = construct_product_system(dfa_ts, ts)
    # show product_system
    from pm4py.objects.petri_net.stochastic.stochastic_transition_system import visualizer as ts_visualizer
    gviz = ts_visualizer.apply(product_system, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
    ts_visualizer.view(gviz)
    unknowns = symbols([f'x_{state.name}' for state in product_system.states])
    equations = get_system_equations(product_system, dfa_ts, ts)
    # Solve the system of equations
    solution = solve(equations, unknowns, dict=True)

    for variable, value in solution[0].items():
        # print(f"{variable} = {value}")
        for product_state in product_system.states:
            if not product_state.incoming:
                if product_state.name in str(variable):
                    probability = value
    return probability

def compute_traces_probability(log,spn,im):
    """
    Compute the probabilities of traces in a log using a Stochastic Petri Net (SPN).

    Parameters:
    - log: Input event log
    - spn: Stochastic Petri Net (SPN)
    - im: Initial marking

    Returns:
    - Dictionary containing probabilities for each trace
    """
    ts = convert.construct_reachability_graph(spn, im)
    # show ts
    from pm4py.objects.petri_net.stochastic.stochastic_transition_system import visualizer as ts_visualizer
    gviz = ts_visualizer.apply(ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
    ts_visualizer.view(gviz)
    variants = {}
    for trace in log:
        # Extract activities from the trace
        activities = tuple(extract_activities_from_trace(trace))
        if activities in variants:
            continue
        net, im, fm = construct_petri_net_from_trace(trace)
        dfa_ts = pm4py.convert_to_reachability_graph(net, im, fm)
        dfa_ts = add_silent_tran_to_reachability_graph(dfa_ts)
        # show dfa_ts
        from pm4py.visualization.transition_system import visualizer as ts_visualizer
        gviz = ts_visualizer.apply(dfa_ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
        ts_visualizer.view(gviz)
        product_system = construct_product_system(dfa_ts, ts)
        # show product_system
        from pm4py.objects.petri_net.stochastic.stochastic_transition_system import visualizer as ts_visualizer
        gviz = ts_visualizer.apply(product_system, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
        ts_visualizer.view(gviz)
        unknowns = symbols([f'x_{state.name}' for state in product_system.states])
        print(f"unknowns: {unknowns}")
        equations = get_system_equations(product_system, dfa_ts, ts)
        # Solve the system of equations
        solution = solve(equations, unknowns, dict=True)
        
        for variable, value in solution[0].items():
            #print(f"{variable} = {value}")
            for product_state in product_system.states:
                if not product_state.incoming:
                    if product_state.name in str(variable):
                        variants[activities] = value
                        print(f"The probability of the trace {activities} is {value}")
    return variants
