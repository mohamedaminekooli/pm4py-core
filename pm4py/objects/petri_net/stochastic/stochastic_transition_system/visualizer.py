from pm4py.visualization.common import gview
from pm4py.visualization.common import save as gsave
from enum import Enum
from pm4py.util import exec_utils, constants
from pm4py.visualization.common.gview import serialize, serialize_dot
from typing import Optional, Dict, Any, Union
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.obj import StochasticTransitionSystem
import tempfile
from copy import copy
import graphviz


def apply(stsys: StochasticTransitionSystem, parameters: Optional[Dict[Any, Any]] = None) -> graphviz.Digraph:
    """
    Get visualization of a Stochastic Transition System

    Parameters
    -----------
    stsys
        Stochastic Transition system
    parameters
        Parameters of the algorithm

    Returns
    ----------
    gviz
        Graph visualization
    """
    return apply_VIEW_BASED(stsys, parameters=parameters)


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


def matplotlib_view(gviz: graphviz.Digraph, parameters=None):
    """
    Views the diagram using Matplotlib

    Parameters
    ---------------
    gviz
        Graphviz
    """

    return gview.matplotlib_view(gviz, parameters=parameters)

#----------------------------VIEW_BASED------------------------------------

class Parameters(Enum):
    FORMAT = "format"
    SHOW_LABELS = "show_labels"
    SHOW_NAMES = "show_names"
    FORCE_NAMES = "force_names"
    FILLCOLORS = "fillcolors"
    FONT_SIZE = "font_size"


def apply_VIEW_BASED(stsys: StochasticTransitionSystem, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> graphviz.Digraph:
    """
    Get visualization of a Stochastic Transition System

    Parameters
    -----------
    stsys
        Stochastic Transition system
    parameters
        Optional parameters of the algorithm

    Returns
    ----------
    gviz
        Graph visualization
    """

    gviz = visualize(stsys, parameters=parameters)
    return gviz


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

class Parameters(Enum):
    FORMAT = "format"
    SHOW_LABELS = "show_labels"
    SHOW_NAMES = "show_names"
    FORCE_NAMES = "force_names"
    FILLCOLORS = "fillcolors"
    FONT_SIZE = "font_size"
    BGCOLOR = "bgcolor"
    SHOW_TRANSITION_NAME = "show transition name"


def visualize(ts, parameters=None):
    if parameters is None:
        parameters = {}

    image_format = exec_utils.get_param_value(Parameters.FORMAT, parameters, "png")
    show_labels = exec_utils.get_param_value(Parameters.SHOW_LABELS, parameters, True)
    show_names = exec_utils.get_param_value(Parameters.SHOW_NAMES, parameters, True)
    force_names = exec_utils.get_param_value(Parameters.FORCE_NAMES, parameters, None)
    fillcolors = exec_utils.get_param_value(Parameters.FILLCOLORS, parameters, {})
    font_size = exec_utils.get_param_value(Parameters.FONT_SIZE, parameters, 11)
    font_size = str(font_size)
    bgcolor = exec_utils.get_param_value(Parameters.BGCOLOR, parameters, constants.DEFAULT_BGCOLOR)
    show_transition_name = exec_utils.get_param_value(Parameters.SHOW_TRANSITION_NAME, parameters, True)

    for state in ts.states:
        state.label = state.name

    perc_char = '%'

    if force_names:
        nts = copy(ts)
        for index, state in enumerate(nts.states):
            state.name = state.name + " (%.2f)" % (force_names[state])
            state.label = "%.2f" % (force_names[state] * 100.0)
            state.label = state.label + perc_char
        ts = nts

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    filename.close()

    viz = graphviz.Digraph(ts.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': bgcolor})

    # states
    viz.attr('node')
    for s in ts.states:
        if show_names:
            if s in fillcolors:
                viz.node(str(id(s)), str(s.label), style="filled", fillcolor=fillcolors[s], fontsize=font_size)
            else:
                viz.node(str(id(s)), str(s.label), fontsize=font_size)
        else:
            if s in fillcolors:
                viz.node(str(id(s)), "", style="filled", fillcolor=fillcolors[s], fontsize=font_size)
            else:
                viz.node(str(id(s)), "", fontsize=font_size)
    # arcs
    for t in ts.transitions:
        if show_labels:
            proba = str(round(t.weight, 4))
            label = f"{str(t.name)}: {proba}"
            if not show_transition_name:
                label = f"{str(get_tran_label(str(t)))}: {proba}"
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label=label, fontsize=font_size)
        else:
            viz.edge(str(id(t.from_state)), str(id(t.to_state)))

    viz.attr(overlap='false')

    viz.format = image_format

    return viz

