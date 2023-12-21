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
from pm4py.objects.conversion.log import converter as log_conversion
from pm4py.visualization.common import gview
from pm4py.visualization.common import save as gsave
from pm4py.visualization.petri_net.variants import wo_decoration, alignments, greedy_decoration_performance, \
    greedy_decoration_frequency, token_decoration_performance, token_decoration_frequency
from pm4py.util import exec_utils
from enum import Enum
from pm4py.objects.petri_net.obj import Marking
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import EventLog, EventStream
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.visualization.common.gview import serialize, serialize_dot
import graphviz
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
import pm4py.objects.petri_net.stochastic.visualization.common.visualize as visualize

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

def apply(spn: StochasticPetriNet, initial_marking: Marking = None, log: Union[EventLog, EventStream, pd.DataFrame] = None, 
          aggregated_statistics=None, parameters: Optional[Dict[Any, Any]] = None) -> graphviz.Digraph:
    if parameters is None:
        parameters = {}
    if log is not None:
        if isinstance(log, pd.DataFrame):
            log = dataframe_utils.convert_timestamp_columns_in_df(log)

        log = log_conversion.apply(log, parameters, log_conversion.TO_EVENT_LOG)
    if aggregated_statistics is None:
        if log is not None:
            aggregated_statistics = token_decoration_frequency.get_decorations(log, spn, initial_marking, parameters=parameters,
                                                measure="performance")
    return visualize.apply(spn, initial_marking, parameters=parameters, decorations=aggregated_statistics)


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
