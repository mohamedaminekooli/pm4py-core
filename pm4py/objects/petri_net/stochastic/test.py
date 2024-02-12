import sys
sys.path.append('../pm4py-core') # or using PYTHONPATH in settings.json
import time
import pandas as pd
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
    log = pm4py.read_xes(os.path.join("tests", "input_data", "running-example.xes"))
    #log = pm4py.read_xes(os.path.join("tests", "input_data", "Sepsis Cases - Event Log.xes"))
    #log = pm4py.read_xes(os.path.join("tests", "input_data", "Road_Traffic_Fine_Management_Process.xes"))
    #log = pm4py.read_xes(os.path.join("tests", "input_data", "BPI Challenge 2017.xes"))
    #log = pm4py.read_xes(os.path.join("tests", "input_data", "Hospital Billing - Event Log.xes"))
    #log = pm4py.read_xes(os.path.join("tests", "input_data", "PrepaidTravelCost.xes"))
    #log = pm4py.read_xes(os.path.join("tests", "input_data", "BPI_Challenge_2013_incidents.xes"))
    return log

if __name__ == "__main__":
    log = import_xes()
    net, im, fm = pm4py.discover_petri_net_inductive(log=log, noise_threshold = 0.2)
    # pm4py.view_petri_net(net, im, fm, format="svg")
    
    # try estimators
    spn = abstractfrequency(log, net)
    # spn = rh(log, net)
    # spn = lh(log, net)
    # spn = fork(log, net, im)
    # spn = mean(log, net)
    # spn = align(log, net, im, fm)

    # import spn
    #spn, im = importer_spn.apply(os.path.join("tests", "test_output_data", "spn_example.slpn"))

    # view spn
    pm4py.view_stochastic_petri_net(spn, im, format="png")
    # export spn
    exporter_spn.apply(spn, im, os.path.join("tests", "test_output_data", "spn_example.slpn"))
    
    # trace probabilty
    from pm4py.objects.petri_net.stochastic import trace_probability
    # variant 1
    probabilities = trace_probability.compute_traces_probability(log,spn,im)
    for trace, proba in probabilities.items():
        print(f"The probability of the trace {trace} is {proba}")
    # variant 2
    trace = ('register request', 'examine casually', 'check ticket', 'decide', 'pay compensation')
    probability = trace_probability.compute_proba_from_trace(spn, im, trace)
    print(f"The probability of the trace {trace} is {probability}")
