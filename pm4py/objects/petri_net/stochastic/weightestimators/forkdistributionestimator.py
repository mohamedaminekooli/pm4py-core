from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
import pm4py
from enum import Enum
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util import constants, exec_utils, xes_constants as xes
from typing import Optional, Dict, Any
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.statistics.start_activities.pandas import get as start_activities_get
from pm4py.statistics.end_activities.pandas import get as end_activities_get
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.conversion.log import converter


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY

class FrequencyCalculator:
    def __init__(self, log: EventLog, petrinet: PetriNet, parameters: Optional[Dict[Any, Any]] = None):
        if parameters is None:
            parameters = {}
        self.log = log
        self.pn = petrinet
        self.parameters = parameters
        self.activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, self.parameters, xes.DEFAULT_NAME_KEY)
        self.eventlog = converter.apply(self.log, variant=converter.Variants.TO_EVENT_LOG, parameters=self.parameters)
    
    def scan_log(self, pn: PetriNet):
        activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, self.parameters, xes.DEFAULT_NAME_KEY)
        activities_occurrences = log_attributes.get_attribute_values(self.log, activity_key, parameters=self.parameters)
        for transition in pn.transitions:
            if transition.label not in activities_occurrences:
                activities_occurrences[transition.label] = 1
        return activities_occurrences
    
    def calculate_activity_pair_frequencies(self):
        activity_pair_frequencies = {}
        for trace in self.eventlog:
            activities = [event[self.activity_key] for event in trace]
            for i in range(len(activities) - 1):
                current_activity = activities[i]
                next_activity = activities[i+1]
                activity_pair = (current_activity, next_activity)
                activity_pair_frequencies[activity_pair] = activity_pair_frequencies.get(activity_pair, 0) + 1
        return activity_pair_frequencies

    def calculate_weights_for_places(self, activity_pair_frequencies):
        weights_places = {}
        for place in self.pn.places:
            if place in im:
                for trace in self.eventlog:
                    weights_places[place] = weights_places.get(place, 0) + 1
            else:
                input_transitions = [arc.source for arc in place.in_arcs]
                output_transitions = [arc.target for arc in place.out_arcs]
                total_activity_pair_frequency = 0
                for input_transition in input_transitions:
                    for output_transition in output_transitions:
                        input_activity = input_transition.label
                        output_activity = output_transition.label
                        total_activity_pair_frequency += activity_pair_frequencies.get((input_activity, output_activity), 0)
                weights_places[place] = max(1,total_activity_pair_frequency)
        return weights_places
    
    def calculate_weights_for_transitions(self, place_weights):
        weight_fork = {}
        activity_frequency = self.scan_log(self.pn)
        for transition in self.pn.transitions:
            input_places = [arc.source for arc in transition.in_arcs]
            for place in input_places:
                if place in place_weights:
                    output_transitions = [arc.target for arc in place.out_arcs]
                    sum_output_transitions = sum(activity_frequency[tran.label] for tran in output_transitions)
                    if transition in weight_fork:
                        weight_fork[transition] += place_weights[place]*activity_frequency[transition.label]/sum_output_transitions
                    else:
                        weight_fork[transition] = place_weights[place]*activity_frequency[transition.label]/sum_output_transitions
        return weight_fork
        
        
class ActivityPairRHWeightEstimator:
    def __init__(self):
        self.place_weights = {}
        self.activities_weights = {}

    def estimate_weights_apply(self, log: EventLog, pn: PetriNet, parameters: Optional[Dict[Any, Any]] = None):
        frequency_calculator = FrequencyCalculator(log, pn)
        activity_pair_frequencies = frequency_calculator.calculate_activity_pair_frequencies()
        place_weights = frequency_calculator.calculate_weights_for_places(activity_pair_frequencies)
        spn = StochasticPetriNet(pn)
        spn = pn
        self.activities_weights = frequency_calculator.calculate_weights_for_transitions(place_weights)
        return self.estimate_activity_pair_weights(spn)

    # Assign weights to transitions based on event frequencies
    def estimate_activity_pair_weights(self, spn: StochasticPetriNet):
        for transition in spn.transitions:
            weight = self.load_activities_weights(transition)
            transition.weight = weight
        return spn

    # Retrieve the frequency of a specific activity
    def load_activities_weights(self, tran):
        # Use a default value of 0.0 if the activity is not found in the log
        frequency = self.activities_weights[tran]
        return frequency


from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple, Set
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace, Event, EventStream
from pm4py.utils import get_properties, __event_log_deprecation_warning
from pm4py.util.pandas_utils import check_is_pandas_dataframe, check_pandas_dataframe_columns
import os
from pm4py import util as pmutil
from pm4py.util import xes_constants as xes_util
from pm4py.objects.conversion.log import converter as log_converter

def discover_stochastic_petrinet_fork_distribution_weight_estimator(log: Union[EventLog, pd.DataFrame, EventStream], 
petri_net: PetriNet, parameters: Optional[Dict[Union[str, Parameters], Any]] = None ) -> StochasticPetriNet:
    if parameters is None:
        parameters = {}
    case_id_glue = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, pmutil.constants.CASE_CONCEPT_NAME)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_util.DEFAULT_NAME_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters, xes_util.DEFAULT_TIMESTAMP_KEY)

    if type(log) not in [pd.DataFrame, EventLog, EventStream]:
        raise Exception("the method can be applied only to a traditional event log!")
    __event_log_deprecation_warning(log)
    if type(log) is not pd.DataFrame:
        log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME, parameters=parameters)
    if check_is_pandas_dataframe(log):
        check_pandas_dataframe_columns(log, activity_key=activity_key, timestamp_key=timestamp_key,
                                        case_id_key=case_id_glue)
    discoverer = ActivityPairRHWeightEstimator()
    return discoverer.estimate_weights_apply(log, petri_net)
    
def use_inductive_miner_petrinet_discovery(log):
    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=0.2)
    pm4py.view_process_tree(tree, format="svg")
    net, im, fm = pm4py.convert_to_petri_net(tree)
    return net, im, fm

#----------------------------------------------------------------------------------------------------------

import tempfile

from graphviz import Digraph

from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net import properties as petri_properties
from pm4py.util import exec_utils, constants
from enum import Enum
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY, PARAMETER_CONSTANT_TIMESTAMP_KEY, DEFAULT_ARTIFICIAL_START_ACTIVITY, DEFAULT_ARTIFICIAL_END_ACTIVITY, STOCHASTIC_DISTRIBUTION


class Parameters_view(Enum):
    FORMAT = "format"
    DEBUG = "debug"
    RANKDIR = "set_rankdir"
    ACTIVITY_KEY = PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = PARAMETER_CONSTANT_TIMESTAMP_KEY
    AGGREGATION_MEASURE = "aggregationMeasure"
    FONT_SIZE = "font_size"
    BGCOLOR = "bgcolor"
    DECORATIONS = "decorations"


def apply_algo(net:StochasticPetriNet, initial_marking, decorations=None, parameters=None):
    """
    Apply method for Petri net visualization (it calls the
    graphviz_visualization method)

    Parameters
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    decorations
        Decorations for elements in the Petri net
    parameters
        Algorithm parameters

    Returns
    -----------
    viz
        Graph object
    """
    if parameters is None:
        parameters = {}

    image_format = exec_utils.get_param_value(Parameters_view.FORMAT, parameters, "png")
    debug = exec_utils.get_param_value(Parameters_view.DEBUG, parameters, False)
    set_rankdir = exec_utils.get_param_value(Parameters_view.RANKDIR, parameters, None)
    font_size = exec_utils.get_param_value(Parameters_view.FONT_SIZE, parameters, "12")
    bgcolor = exec_utils.get_param_value(Parameters_view.BGCOLOR, parameters, constants.DEFAULT_BGCOLOR)

    if decorations is None:
        decorations = exec_utils.get_param_value(Parameters_view.DECORATIONS, parameters, None)
    return graphviz_visualization(net, image_format=image_format, initial_marking=initial_marking,
                                  decorations=decorations, debug=debug,
                                  set_rankdir=set_rankdir, font_size=font_size, bgcolor=bgcolor)


def graphviz_visualization(net:StochasticPetriNet, image_format="png", initial_marking=None, decorations=None,
                           debug=False, set_rankdir=None, font_size="12", bgcolor=constants.DEFAULT_BGCOLOR):
    """
    Provides visualization for the petrinet

    Parameters
    ----------
    net: :class:`pm4py.entities.petri.petrinet.PetriNet`
        Petri net
    image_format
        Format that should be associated to the image
    initial_marking
        Initial marking of the Petri net
    decorations
        Decorations of the Petri net (says how element must be presented)
    debug
        Enables debug mode
    set_rankdir
        Sets the rankdir to LR (horizontal layout)

    Returns
    -------
    viz :
        Returns a graph object
    """
    if initial_marking is None:
        initial_marking = Marking()
    if decorations is None:
        decorations = {}

    font_size = str(font_size)

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    filename.close()

    viz = Digraph(net.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': bgcolor})
    if set_rankdir:
        viz.graph_attr['rankdir'] = set_rankdir
    else:
        viz.graph_attr['rankdir'] = 'LR'

    # transitions
    viz.attr('node', shape='box')
    for t in net.transitions:
        label = decorations[t]["label"] if t in decorations and "label" in decorations[t] else ""
        fillcolor = decorations[t]["color"] if t in decorations and "color" in decorations[t] else None
        textcolor = "black"

        if t.label is not None and not label:
            label = t.label
        if debug:
            label = t.name
        label = str(label)

        if fillcolor is None:
            if t.label is None:
                fillcolor = "black"
                if label:
                    textcolor = "white"
            else:
                fillcolor = bgcolor

        # Add transition weight to the label
        weight = decorations[t]["weight"] if t in decorations and "weight" in decorations[t] else 1
        if t.weight is not None:
            weight = t.weight
        if debug:
            weight = t.weight
        weight = str(weight)
        label += f" ({weight})"
        if t.label is None:
            textcolor = "white"

        viz.node(str(id(t)), label, style='filled', fillcolor=fillcolor, border='1', fontsize=font_size, fontcolor=textcolor)

        if petri_properties.TRANS_GUARD in t.properties:
            guard = t.properties[petri_properties.TRANS_GUARD]
            viz.node(str(id(t))+"guard", style="dotted", label=guard)
            viz.edge(str(id(t))+"guard", str(id(t)), arrowhead="none", style="dotted")

    # places
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted([x for x in list(net.places) if x in initial_marking], key=lambda x: x.name)
    places_sort_list_rest = sorted([x for x in list(net.places) if not x in initial_marking],
                                 key=lambda x: x.name)
    places_sort_list_not_im_rest = sorted(
        [x for x in list(net.places) if x not in initial_marking], key=lambda x: x.name)
    # making the addition happen in this order:
    # - first, the places belonging to the initial marking
    # - after, the places not belonging to the initial marking
    # - at last, the places not belonging to the initial marking
    # in this way, is more probable that the initial marking is on the left and the rest on the right
    places_sort_list = places_sort_list_im + places_sort_list_not_im_rest + places_sort_list_rest

    for p in places_sort_list:
        label = decorations[p]["label"] if p in decorations and "label" in decorations[p] else ""
        fillcolor = decorations[p]["color"] if p in decorations and "color" in decorations[p] else bgcolor

        label = str(label)
        if p in initial_marking:
            if initial_marking[p] == 1:
                viz.node(str(id(p)), "<&#9679;>", fontsize="34", fixedsize='true', shape="circle", width='0.75', style="filled", fillcolor=fillcolor)
            else:
                viz.node(str(id(p)), str(initial_marking[p]), fontsize="34", fixedsize='true', shape="circle", width='0.75', style="filled", fillcolor=fillcolor)
        else:
            if debug:
                viz.node(str(id(p)), str(p.name), fontsize=font_size, shape="ellipse")
            else:
                if p in decorations and "color" in decorations[p] and "label" in decorations[p]:
                    viz.node(str(id(p)), label, style='filled', fillcolor=fillcolor,
                             fontsize=font_size, shape="ellipse")
                else:
                    viz.node(str(id(p)), label, shape='circle', fixedsize='true', width='0.75', style="filled", fillcolor=fillcolor)

    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))

    # check if there is an arc with weight different than 1.
    # in that case, all the arcs in the visualization should have the arc weight visible
    arc_weight_visible = False
    for arc in arcs_sort_list:
        if arc.weight != 1:
            arc_weight_visible = True
            break

    for a in arcs_sort_list:
        penwidth = decorations[a]["penwidth"] if a in decorations and "penwidth" in decorations[a] else None
        label = decorations[a]["label"] if a in decorations and "label" in decorations[a] else ""
        color = decorations[a]["color"] if a in decorations and "color" in decorations[a] else None

        if not label and arc_weight_visible:
            label = a.weight

        label = str(label)
        arrowhead = "normal"

        if petri_properties.ARCTYPE in a.properties:
            if a.properties[petri_properties.ARCTYPE] == petri_properties.RESET_ARC:
                arrowhead = "vee"
            elif a.properties[petri_properties.ARCTYPE] == petri_properties.INHIBITOR_ARC:
                arrowhead = "dot"

        viz.edge(str(id(a.source)), str(id(a.target)), label=label,
                 penwidth=penwidth, color=color, fontsize=font_size, arrowhead=arrowhead, fontcolor=color)

    viz.attr(overlap='false')

    viz.format = image_format.replace("html", "plain-ext")

    return viz


def view_stochastic_petri_net(petri_net: StochasticPetriNet, initial_marking: Optional[Marking] = None,
                   format: str = constants.DEFAULT_FORMAT_GVIZ_VIEW, bgcolor: str = "white",
                   decorations: Dict[Any, Any] = None, debug: bool = False, rankdir: str = constants.DEFAULT_RANKDIR_GVIZ):
    format = str(format).lower()
    #from pm4py.visualization.petri_net import visualizer as pn_visualizer
    gviz = apply(petri_net, initial_marking, parameters={Variants.WO_DECORATION.value.Parameters.FORMAT: format,
                "bgcolor": bgcolor, "decorations": decorations, "debug": debug, "set_rankdir": rankdir})
    view(gviz)
from pm4py.objects.conversion.log import converter as log_conversion
from pm4py.visualization.common import gview
from pm4py.visualization.common import save as gsave
from pm4py.visualization.petri_net.variants import wo_decoration, alignments, greedy_decoration_performance, \
    greedy_decoration_frequency, token_decoration_performance, token_decoration_frequency
from pm4py.util import exec_utils
from enum import Enum
from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import EventLog, EventStream
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.visualization.common.gview import serialize, serialize_dot
import graphviz
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet


class Variants(Enum):
    WO_DECORATION = wo_decoration
    FREQUENCY = token_decoration_frequency
    PERFORMANCE = token_decoration_performance
    FREQUENCY_GREEDY = greedy_decoration_frequency
    PERFORMANCE_GREEDY = greedy_decoration_performance
    ALIGNMENTS = alignments


WO_DECORATION = Variants.WO_DECORATION
FREQUENCY_DECORATION = Variants.FREQUENCY
PERFORMANCE_DECORATION = Variants.PERFORMANCE
FREQUENCY_GREEDY = Variants.FREQUENCY_GREEDY
PERFORMANCE_GREEDY = Variants.PERFORMANCE_GREEDY
ALIGNMENTS = Variants.ALIGNMENTS


def apply(net: StochasticPetriNet, initial_marking: Marking = None, log: Union[EventLog, EventStream, pd.DataFrame] = None, aggregated_statistics=None, parameters: Optional[Dict[Any, Any]] = None,
          variant=Variants.WO_DECORATION) -> graphviz.Digraph:
    if parameters is None:
        parameters = {}
    if log is not None:
        if isinstance(log, pd.DataFrame):
            log = dataframe_utils.convert_timestamp_columns_in_df(log)

        log = log_conversion.apply(log, parameters, log_conversion.TO_EVENT_LOG)
    return apply_viz(net, initial_marking, log=log,
                                                 aggregated_statistics=aggregated_statistics,
                                                 parameters=parameters)

def apply_viz(net: PetriNet, initial_marking: Marking, log: EventLog = None, aggregated_statistics=None, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> graphviz.Digraph:
    """
    Apply method for Petri net visualization (it calls the graphviz_visualization
    method) adding performance representation obtained by token replay

    Parameters
    -----------
    net
        Petri net
    initial_marking
        Initial marking
    log
        (Optional) log
    aggregated_statistics
        Dictionary containing the frequency statistics
    parameters
        Algorithm parameters (including the activity key used during the replay, and the timestamp key)

    Returns
    -----------
    viz
        Graph object
    """
    if aggregated_statistics is None:
        if log is not None:
            aggregated_statistics = token_decoration_frequency.get_decorations(log, net, initial_marking, parameters=parameters,
                                                    measure="performance")
    return apply_algo(net, initial_marking, parameters=parameters,
                           decorations=aggregated_statistics)

def save(gviz: graphviz.Digraph, output_file_path: str, parameters=None):
    """
    Save the diagram

    Parameters
    -----------
    gviz
        GraphViz diagram
    output_file_path
        Path where the GraphViz output should be saved
    """
    gsave.save(gviz, output_file_path, parameters=parameters)


def view(gviz: graphviz.Digraph, parameters=None):
    """
    View the diagram

    Parameters
    -----------
    gviz
        GraphViz diagram
    """
    return gview.view(gviz, parameters=parameters)

# Usage example:
log = pm4py.read_xes(os.path.join("..", "tests", "input_data", "example_12.xes"))
net, im, fm = use_inductive_miner_petrinet_discovery(log)
spn = discover_stochastic_petrinet_fork_distribution_weight_estimator(log, net)
view_stochastic_petri_net(spn, im, format="svg")
