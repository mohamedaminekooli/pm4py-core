import pm4py


from pm4py.statistics.variants.log import get as variants_module
log = pm4py.convert_to_event_log(log)
language = variants_module.get_language(log)
print(language)
#net, im, fm = pm4py.discover_petri_net_alpha(log)
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
playout_log = simulator.apply(spn, im, parameters={simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.LOG: log},
                                variant=simulator.Variants.STOCHASTIC_PLAYOUT)
print(playout_log)
model_language = variants_module.get_language(playout_log)
print(model_language)

from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
emd = emd_evaluator.apply(model_language, language)
print(emd)
