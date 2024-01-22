#--------------------------------------Get the reachability graph of the SPN, with the probability of transition---------------------------------------
import os
import sys
sys.path.insert(0, 'C:\\Users\\MohamedAmineKooli\\BA\\pm4py-core')
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.stochastic import discover_spn
import pm4py

def abstractfrequency(log, net):
    return discover_spn.discover_stochastic_petrinet_abstract_fraquency_estimator(log, net)

log = xes_importer.apply(os.path.join("tests", "input_data", "example_12.xes"))
net, im, fm = discover_spn.use_inductive_miner_petrinet_discovery(log)
spn = abstractfrequency(log, net)
#pm4py.view_stochastic_petri_net(spn, im, format="svg")

from pm4py.objects.petri_net.stochastic.utils import convert

ts = convert.construct_reachability_graph(spn, im)

from pm4py.objects.petri_net.stochastic.stochastic_transition_system import visualizer as ts_visualizer

gviz = ts_visualizer.apply(ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
ts_visualizer.view(gviz)

#---------------------------------For each trace, get a DFA that accepts this trace, with silenced transitions so----------------------
#---------------------------------that it accepts infinitely many runs that can project to the trace-------------------------------------

import os
from enum import Enum
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.objects import petri_net
# from pm4py.objects.transition_system import obj as ts
from pm4py.objects.transition_system import utils
from pm4py.util import exec_utils

log = xes_importer.apply(os.path.join("tests", "input_data", "example_12.xes"))

def extract_activities_from_trace(trace):
    return [event["concept:name"] for event in trace]

def construct_petri_net_from_trace(trace):
    net, initial_marking, final_marking = petri_net.utils.petri_utils.construct_trace_net(trace)
    return net, initial_marking, final_marking

#for trace in log:
# Extract activities from the trace
#activities = extract_activities_from_trace(trace)

from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY, PARAMETER_CONSTANT_CASEID_KEY, CASE_CONCEPT_NAME
from pm4py.util.xes_constants import DEFAULT_NAME_KEY, DEFAULT_TRACEID_KEY


activity_key = exec_utils.get_param_value(PARAMETER_CONSTANT_ACTIVITY_KEY, parameters=None, default=DEFAULT_NAME_KEY)
from pm4py.objects.log.obj import Trace, Event
trace=('a','b','c','b','d')
case=Trace()
for act in trace:
    case.append(Event({activity_key: act}))
# Create a Petri net from the trace
net, im, fm = construct_petri_net_from_trace(case)
#pm4py.view_petri_net(net, im, fm, format="svg")
# Construct a reachability graph (DFA) from the Petri net
dfa_ts = pm4py.convert_to_reachability_graph(net, im, fm)
for state in dfa_ts.states:
    utils.add_arc_from_to(name=str(("skip", None)), fr=state, to=state, ts=dfa_ts)
from pm4py.visualization.transition_system import visualizer as ts_visualizer
# Visualize the transition system ts
gviz = ts_visualizer.apply(dfa_ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
ts_visualizer.view(gviz)

#---------------------Construct a cross-product of DFA and reachability graph, so that you can write an equations system.---------------------------
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.stochastic_transition_system import StochasticTransitionSystem

def get_tran_label(tran):
    comma_index = tran.find(',')
    closing_parenthesis_index = tran.find(')')
    # Extract the substring between ',' and ')'
    dfa_tran_name = tran[comma_index + 1: closing_parenthesis_index].strip()
    return dfa_tran_name
def construct_product_system(dfa_ts, spn_ts):
    """
    Construct the product system of a DFA and a Stochastic Transition System.

    Parameters:
    - dfa_ts: Transition system of the Deterministic Finite Automaton (DFA)
    - spn_ts: Transition system of the Stochastic Petri Net (SPN)

    Returns:
    - product_system: StochasticTransitionSystem representing the product system
    """
    #--------------------------------------------------
    product_system = StochasticTransitionSystem()
    dfa_im = []
    dfa_fm = []
    spn_im = []
    spn_fm = []
    product_im=[]
    product_fm=[]
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
    # print(f"dfa_im: {dfa_im}")
    # print(f"dfa_fm: {dfa_fm}")
    # print(f"spn_im: {spn_im}")
    # print(f"spn_fm: {spn_fm}")
    for spn_state in spn_ts.states:      
        for dfa_state in dfa_ts.states:
            # Combine DFA state and SPN state to form a new state in the product system
            product_state_name = f"{spn_state.name};{dfa_state.name}"
            product_state = StochasticTransitionSystem.State(name = product_state_name)
            # Check if it is the initial state, and set as the initial state of the product system
            for s_im in spn_im:
                for d_im in dfa_im:
                    if s_im.name in product_state_name and d_im.name in product_state_name:
                        product_im.append(product_state)
            
            for fm in spn_fm + dfa_fm:
                if fm.name in product_state_name:
                    product_fm.append(product_state)
            
            if spn_state in spn_im and dfa_state in dfa_im:
                # fix initial state
                product_state.outgoing = spn_state.outgoing
                product_system.states.add(product_state)
                map_next_state=[product_state]
                print(f"map_next_state: {map_next_state}")
                i=0
                while i< len(map_next_state):
                    print(len(map_next_state))
                    product_state = map_next_state[i]
                    print(f"product_state: {product_state}")
                    for spn_state in spn_ts.states:
                        if spn_state.name == product_state.name.split(';')[0]:
                            product_state.outgoing = spn_state.outgoing
                    for out_transition in product_state.outgoing:
                        product_transition_name = f"{out_transition.name}"
                        print(f"product_transition: {product_transition_name}")
                        product_transition = StochasticTransitionSystem.Transition(
                            name=product_transition_name,
                            from_state= product_state,
                            data=out_transition.data,
                            weight=out_transition.weight
                        )
                        for current_dfa_state in dfa_ts.states:
                            if current_dfa_state.name == product_state.name.split(';')[1]:
                                dfa_state=current_dfa_state
                        exist=False
                        # dfa_tran_next = None
                        for dfa_tran in dfa_state.outgoing:
                            if get_tran_label(str(dfa_tran))=='None':
                                dfa_tran_none = dfa_tran
                            if get_tran_label(str(dfa_tran)) == get_tran_label(product_transition_name):
                                exist=True
                                for tran_dfa in dfa_ts.transitions:
                                    if tran_dfa.name==dfa_tran.name and tran_dfa.from_state == dfa_state:
                                        dfa_tran_next = tran_dfa
                                next_state_name = f"{out_transition.to_state};{dfa_tran_next.to_state}"
                                for state in product_system.states:
                                    if state.name == str(next_state_name):
                                        print('next already exists')
                                        next_state=state
                                        break
                                    else:
                                        next_state = StochasticTransitionSystem.State(name = next_state_name)
                                #add product transition
                                product_transition.to_state = next_state
                                add_tr=True
                                for tr in product_system.transitions:
                                    if tr.name==product_transition.name and tr.from_state==product_transition.from_state:
                                        add_tr=False
                                        break
                                if add_tr:
                                    product_system.transitions.add(product_transition)
                                print(f"next_state_name: {next_state_name}")
                                next_state.incoming.add(product_transition)
                                if next_state not in product_system.states:
                                    product_system.states.add(next_state)
                                break
                        if not exist:
                            next_state_name = f"{out_transition.to_state};{dfa_tran_none.to_state}"
                            for state in product_system.states:
                                if state.name == str(next_state_name):
                                    print('next already exists')
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
                            print(f"next_state_name: {next_state_name}")
                            next_state.incoming.add(product_transition)
                            if next_state not in product_system.states:
                                    product_system.states.add(next_state)
                        
                        if next_state not in map_next_state:
                            map_next_state.append(next_state)
                    i+=1
                    print(f"map_next_state: {map_next_state}")

    return product_system


# Example usage:
product_system = construct_product_system(dfa_ts, ts)

from pm4py.objects.petri_net.stochastic.stochastic_transition_system import visualizer as ts_visualizer

gviz = ts_visualizer.apply(product_system, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
ts_visualizer.view(gviz)


from sympy import symbols, Eq, solve
def calculate_x(product_system, unknowns):
    x={}
    for dfa_state in dfa_ts.states:
        if len(dfa_state.outgoing) == 1:
            final_dfa_state_name = dfa_state.name
    print(unknowns)
    equations = []
    for state in product_system.states:
        x_state = symbols(f'x_{state.name}')
        if not state.outgoing:
            if final_dfa_state_name == state.name.split(';')[1]:
                equation = Eq(x_state, 1.0)
            else:
                equation = Eq(x_state, 0.0)
        else:
            val=0.0
            for tran in state.outgoing:
                weight=tran.weight
                for tr in product_system.transitions:
                    if tr.name==tran.name and str(tr.from_state)==state.name:
                        to_st=tr.to_state
                #print(f"{tran}={weight}")
                x = symbols(f'x_{to_st.name}')
                val+=weight*x
            print(f"{x_state}={val}")
            equation = Eq(x_state, val)
        equations.append(equation)
    return equations

unknowns = symbols([f'x_{state.name}' for state in product_system.states])
equations = calculate_x(product_system,unknowns)
print(equations)
# Solve the system of equations
solution = solve(equations, unknowns, dict=True)

print("Solution:")
for variable, value in solution[0].items():
    print(f"{variable} = {value}")
    for product_state in product_system.states:
        if not product_state.incoming:
            if product_state.name in str(variable):
                print(f"The probability of the trace {trace} is {value}")