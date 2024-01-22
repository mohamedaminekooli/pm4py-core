import os
import xml.etree.ElementTree as ET
import random

def create_xes_file(input_file, output_file, num_traces):
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Get all traces from the original file
    all_traces = root.findall('.//trace')

    # Randomly select the desired number of traces
    selected_traces = random.sample(all_traces, min(num_traces, len(all_traces)))

    # Create a list of elements to remove
    elements_to_remove = [trace for trace in all_traces if trace not in selected_traces]

    # Remove the elements from the root
    for element in elements_to_remove:
        root.remove(element)

    # Save the new file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
def apply():
    # Specify input and output file paths
    input_file_path = os.path.join("tests", "input_data", "example_12.xes")
    output_file_1_trace = os.path.join("pm4py", "objects", "petri_net", "stochastic", "output_file_1_traces.xes")
    output_file_10_traces = os.path.join("pm4py", "objects", "petri_net", "stochastic", "output_file_10_traces.xes")
    output_file_100_traces = os.path.join("pm4py", "objects", "petri_net", "stochastic", "output_file_100_traces.xes")
    output_file_1000_traces = os.path.join("pm4py", "objects", "petri_net", "stochastic", "output_file_1000_traces.xes")
    output_file_10000_traces = os.path.join("pm4py", "objects", "petri_net", "stochastic", "output_file_10000_traces.xes")
    output_file_100000_traces = os.path.join("pm4py", "objects", "petri_net", "stochastic", "output_file_100000_traces.xes")

    # Create files with different numbers of traces
    create_xes_file(input_file_path, output_file_1_trace, 1)
    create_xes_file(input_file_path, output_file_10_traces, 10)
    create_xes_file(input_file_path, output_file_100_traces, 100)
    create_xes_file(input_file_path, output_file_1000_traces, 1000)
    create_xes_file(input_file_path, output_file_10000_traces, 10000)
    create_xes_file(input_file_path, output_file_100000_traces, 100000)
