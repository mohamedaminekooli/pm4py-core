'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
from pm4py.visualization.common import gview
from pm4py.visualization.common import save as gsave
from pm4py.visualization.transition_system.variants import view_based, trans_frequency
from enum import Enum
from pm4py.util import exec_utils
from pm4py.visualization.common.gview import serialize, serialize_dot
from typing import Optional, Dict, Any
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.stochastic_transition_system import StochasticTransitionSystem

import graphviz


class Variants(Enum):
    VIEW_BASED = view_based
    TRANS_FREQUENCY = trans_frequency


DEFAULT_VARIANT = Variants.VIEW_BASED


def apply(stsys: StochasticTransitionSystem, parameters: Optional[Dict[Any, Any]] = None, variant=DEFAULT_VARIANT) -> graphviz.Digraph:
    """
    Get visualization of a Stochastic Transition System

    Parameters
    -----------
    stsys
        Stochastic Transition system
    parameters
        Parameters of the algorithm
    variant
        Variant of the algorithm to use, including:
            - Variants.VIEW_BASED

    Returns
    ----------
    gviz
        Graph visualization
    """
    # return exec_utils.get_variant(variant).apply(stsys, parameters=parameters)
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

#--------------------------TRANS_FREQUENCY----------------------------------

'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import uuid

from typing import Optional, Dict, Any, Union
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.stochastic_transition_system import StochasticTransitionSystem
import graphviz

import tempfile

from graphviz import Digraph
from pm4py.util import exec_utils, constants
from enum import Enum


class Parameters(Enum):
    FORMAT = "format"
    BGCOLOR = "bgcolor"


def get_perc(total_events, arc_events):
    if total_events > 0:
        return " " + str(total_events) + " / %.2f %%" % (100.0 * arc_events / total_events)
    return " 0 / 0.00 %"


def apply_TRANS_FREQUENCY(tsys: StochasticTransitionSystem, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> graphviz.Digraph:
    if parameters is None:
        parameters = {}

    image_format = exec_utils.get_param_value(Parameters.FORMAT, parameters, "png")
    bgcolor = exec_utils.get_param_value(Parameters.BGCOLOR, parameters, constants.DEFAULT_BGCOLOR)

    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    filename.close()

    viz = Digraph(tsys.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': bgcolor})

    states_dictio = {}

    for s in tsys.states:
        node_uuid = str(uuid.uuid4())
        states_dictio[id(s)] = node_uuid

        sum_ingoing = 0
        sum_outgoing = 0

        for t in s.incoming:
            sum_ingoing += len(t.data["events"])

        for t in s.outgoing:
            sum_outgoing += len(t.data["events"])

        fillcolor = "white"

        if sum_ingoing != len(s.data["ingoing_events"]) or sum_outgoing != len(s.data["outgoing_events"]):
            fillcolor = "red"

        taillabel = get_perc(sum_ingoing, len(s.data["ingoing_events"]))
        headlabel = get_perc(sum_outgoing, len(s.data["outgoing_events"]))

        label = "IN=" + taillabel + "\n" + str(s.name) + "\nOUT=" + headlabel

        viz.node(node_uuid, label=label, fontsize="10", style="filled", fillcolor=fillcolor)

    for t in tsys.transitions:
        proba = str(round(t.weight, 4))
        label = f"{str(t.name)}: {proba}"
        viz.edge(states_dictio[id(t.from_state)], states_dictio[id(t.to_state)], fontsize="8", label=label,
                 taillabel=get_perc(len(t.from_state.data["outgoing_events"]), len(t.data["events"])),
                 headlabel=get_perc(len(t.to_state.data["ingoing_events"]), len(t.data["events"])))

    viz.attr(overlap='false')

    viz.format = image_format.replace("html", "plain-ext")

    return viz

#----------------------------VIEW_BASED-------------------------------------------

'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
from pm4py.visualization.transition_system.util import visualize_graphviz
from enum import Enum
from typing import Optional, Dict, Any, Union
from pm4py.objects.petri_net.stochastic.stochastic_transition_system.stochastic_transition_system import StochasticTransitionSystem
import graphviz


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

#------------------
'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import tempfile
from copy import copy

from graphviz import Digraph
from pm4py.util import exec_utils, constants
from enum import Enum


class Parameters(Enum):
    FORMAT = "format"
    SHOW_LABELS = "show_labels"
    SHOW_NAMES = "show_names"
    FORCE_NAMES = "force_names"
    FILLCOLORS = "fillcolors"
    FONT_SIZE = "font_size"
    BGCOLOR = "bgcolor"


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

    viz = Digraph(ts.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': bgcolor})

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
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label=label, fontsize=font_size)
        else:
            viz.edge(str(id(t.from_state)), str(id(t.to_state)))

    viz.attr(overlap='false')

    viz.format = image_format

    return viz

