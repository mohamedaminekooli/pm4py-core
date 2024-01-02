import sys

import pandas as pd
sys.path.insert(0, 'C:\\Users\\nader\\OneDrive\\Bureau\\existing_pm4py\\pm4py-core')
import pm4py
from pm4py.objects.petri_net.stochastic import discover_spn
import os
from pm4py.objects.petri_net.stochastic.importer import importer_spn


def abstractfrequency(log, net):
    return discover_spn.discover_stochastic_petrinet_abstract_fraquency_estimator(log, net)
def rh(log, net):
    return discover_spn.discover_stochastic_petrinet_activity_pair_rh_weight_estimator(log, net)
def lh(log, net):
    return discover_spn.discover_stochastic_petrinet_activity_pair_lh_weight_estimator(log, net)
def fork(log, net, im):
    return discover_spn.discover_stochastic_petrinet_forkdistributionestimator(log, net, im)
def mean(log, net):
    return discover_spn.discover_stochastic_petrinet_meanscaledactivitypairfrequencyestimator(log, net)
def align(log, net, im, fm):
    return discover_spn.discover_stochastic_petrinet_alignmentestimator(log, net, im, fm)

def import_xes():
    log = pm4py.read_xes(os.path.join("tests", "input_data", "example_12.xes"))
    return log
def import_csv():
    log = pd.read_csv(os.path.join("tests", "input_data", "roadtraffic100traces.csv"), sep=',')
    log = pm4py.format_dataframe(log, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    return log
log = import_xes()
net, im, fm = discover_spn.use_inductive_miner_petrinet_discovery(log)
pm4py.view_petri_net(net, im, fm, format="svg")

#spn = abstractfrequency(log, net)
#spn = rh(log, net)
#spn = lh(log, net)
#spn = fork(log, net, im)
#spn = mean(log, net)
spn = align(log, net, im, fm)

pm4py.view_stochastic_petri_net(spn, im, format="svg")

#exporter_spn.apply(spn, im, os.path.join("tests", "test_output_data", "example_12.slpn"))

def import_spln_script():
    spn, im = importer_spn.apply(os.path.join("tests", "test_output_data", "example_12.slpn"))
    pm4py.view_stochastic_petri_net(spn, im, format="svg")
    return spn, im
#spn, im = import_spln_script()
