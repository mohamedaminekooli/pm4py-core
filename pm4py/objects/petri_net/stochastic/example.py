import sys

import pandas as pd
sys.path.insert(0, 'C:\\Users\\MohamedAmineKooli\\BA\\pm4py-core')
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
    log = pm4py.read_xes(os.path.join("tests", "input_data", "interval_event_log.xes"))
    #log = pm4py.read_xes(os.path.join("pm4py", "objects", "petri_net", "stochastic", "output_file_100000_traces.xes"))
    return log
def import_csv():
    log = pd.read_csv(os.path.join("tests", "input_data", "receipt.csv"), sep=',')
    log = pm4py.format_dataframe(log, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    return log
log = import_xes()
net, im, fm = discover_spn.use_inductive_miner_petrinet_discovery(log)
#pm4py.view_petri_net(net, im, fm, format="svg")
import time
start_time = time.time()
spn = abstractfrequency(log, net)
#spn = rh(log, net)
#spn = lh(log, net)
#spn = fork(log, net, im)
#spn = mean(log, net)
#spn = align(log, net, im, fm)
end_time = time.time()
execution_time = end_time - start_time
speed = 1 / execution_time if execution_time > 0 else float('inf')
print(f"[{execution_time:.2f}s<{speed:.2f}it/s]")
pm4py.view_stochastic_petri_net(spn, im, format="svg")

#exporter_spn.apply(spn, im, os.path.join("tests", "test_output_data", "example_12.slpn"))

def import_spln_script():
    spn, im = importer_spn.apply(os.path.join("tests", "test_output_data", "example_12.slpn"))
    pm4py.view_stochastic_petri_net(spn, im, format="svg")
    return spn, im
#spn, im = import_spln_script()

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