import csv


def read_channel_list(channel_list_fn):
    # Read label list
    with open(channel_list_fn) as f:
        channel_list = f.readlines()
    return [label.strip().split(':')[0] for label in channel_list]


def get_label_list(connection_fn):
    """Open a list of connections and return a label list
    """
    label_list = []
    with open(connection_fn, 'r') as adj_file:
        for line in adj_file:
            # Get adjacencies
            node, edges = line.strip().split(':')
            node = node.upper()
            edges = edges.upper()
            label_list.append(node)
    return label_list


def read_manifest(manifest_fn):
    """Return a list of ordered dictionaries for each file in a manifest
    """
    with open(manifest_fn, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        dicts = list(reader)
    return dicts


def write_csv(fn, toCSV):
    """Write a list of dictionaries to a csv file"""
    keys = toCSV[0].keys()
    with open(fn, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)
