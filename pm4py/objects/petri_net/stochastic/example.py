import sys
import time
import pandas as pd
sys.path.insert(0, 'C:\\Users\\MohamedAmineKooli\\BA\\pm4py-core')
import pm4py
from pm4py.objects.petri_net.stochastic import discover_spn
import os
from pm4py.objects.petri_net.stochastic.importer import importer_spn
from pm4py.objects.petri_net.stochastic.exporter import exporter_spn


def abstractfrequency(log, net):
    return discover_spn.discover_spn_abstract_fraquency_estimator(log, net)
def rh(log, net):
    return discover_spn.discover_spn_activity_pair_rh_weight_estimator(log, net)
def lh(log, net):
    return discover_spn.discover_spn_activity_pair_lh_weight_estimator(log, net)
def fork(log, net, im):
    return discover_spn.discover_spn_fork_distribution_estimator(log, net, im)
def mean(log, net):
    return discover_spn.discover_spn_mean_scaled_activity_pair_frequency_estimator(log, net)
def align(log, net, im, fm):
    return discover_spn.discover_spn_alignment_estimator(log, net, im, fm)

def import_xes():
    log = pm4py.read_xes(os.path.join("tests", "input_data", "interval_event_log.xes"))
    #log = pm4py.read_xes(os.path.join("pm4py", "objects", "petri_net", "stochastic", "output_file_10_traces.xes"))
    return log
def import_csv():
    log = pd.read_csv(os.path.join("tests", "input_data", "receipt.csv"), sep=',')
    log = pm4py.format_dataframe(log, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    return log
log = import_xes()
net, im, fm = discover_spn.use_inductive_miner_petrinet_discovery(log)
#import pm4py.objects.petri_net.exporter.exporter as export_pn
#export_pn.apply(net, im, output_filename=os.path.join("tests", "test_output_data","test_pn.pnml"), final_marking=fm)
#from pm4py.objects.conversion.log import converter
#log = converter.apply(log, variant=converter.Variants.TO_EVENT_STREAM)
start_time = time.time()
#spn = abstractfrequency(log, net)
#spn = rh(log, net)
#spn = lh(log, net)
spn = fork(log, net, im)
#spn = mean(log, net)
# spn = align(log, net, im, fm)
end_time = time.time()
execution_time = end_time - start_time
print(f"execution_time = {execution_time:.2f}")
pm4py.view_stochastic_petri_net(spn, im, format="svg")

# exporter_spn.apply(spn, im, os.path.join("tests", "test_output_data", "example_12.slpn"))

def import_spln_script():
    spn, im = importer_spn.apply(os.path.join("tests", "input_data", "PetriNet.slpn"))
    return spn, im
#spn, im = import_spln_script()
#pm4py.view_stochastic_petri_net(spn, im, format="svg")

# EMSC
def emsc():
    from pm4py.statistics.variants.log import get as variants_module
    log = pm4py.convert_to_event_log(log)
    language = variants_module.get_language(log)
    #print(language)
    #net, im, fm = pm4py.discover_petri_net_alpha(log)
    from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
    playout_log = simulator.apply(spn, im, parameters={simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.LOG: log},
                                    variant=simulator.Variants.STOCHASTIC_PLAYOUT)
    #print(playout_log)
    model_language = variants_module.get_language(playout_log)
    #print(model_language)

    from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
    emd = emd_evaluator.apply(model_language, language)
    print(emd)

# trace probabilty
# from pm4py.objects.conversion.log import converter as log_converter
# from pm4py.objects.petri_net.stochastic import trace_proba

# log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
# # variant 1
# trace_probability = trace_proba.compute_traces_probability(log,spn,im)
# sum=0
# for trace, proba in trace_probability.items():
#     print(f"The probability of the trace {trace} is {proba}")
#     sum+=proba
# print(f"The sum of probabilities is {sum}")
# variant 2
# trace = ('place order', 'send invoice', 'send reminder', 'pay', 'prepare delivery', 'confirm payment', 'make delivery')
# trace_probability = trace_proba.compute_proba_from_trace(spn, im, trace)
# print(f"The probability of the trace {trace} is {trace_probability}")

import matplotlib.pyplot as plt
import pm4py

def measure_execution_time(estimator_func, log, net, im, fm):
    
    if estimator_func in [abstractfrequency, rh, lh, mean]:
        start_time = time.time()
        spn = estimator_func(log, net)
    elif estimator_func in [fork]:
        start_time = time.time()
        spn = estimator_func(log, net, im)
    elif estimator_func in [align]:
        start_time = time.time()
        spn = estimator_func(log, net, im, fm)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time

# log = import_xes()
# net, im, fm = discover_spn.use_inductive_miner_petrinet_discovery(log)

#-------------------------------------

# estimators = [
#     ("Abstract Frequency", abstractfrequency),
#     ("RH", rh),
#     ("LH", lh),
#     ("Fork", fork),
#     ("Mean", mean),
#     ("Align", align)
# ]

# execution_times = {}

# for estimator_name, estimator_func in estimators:
#     time_taken = measure_execution_time(estimator_func, log, net, im, fm)
#     execution_times[estimator_name] = time_taken
#     print(f"{estimator_name} execution time: {time_taken*1000:.2f} seconds")

# xes_file_name = "Sepsis Cases - Event Log.xes"
# fig, ax = plt.subplots()
# ax.bar(execution_times.keys(), execution_times.values())
# ax.set_ylabel('Execution Time (seconds)')
# ax.set_title(f'Execution Time of SPN Estimators for {xes_file_name}')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()
