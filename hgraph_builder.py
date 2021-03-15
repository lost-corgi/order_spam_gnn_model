"""Graph builder from pandas dataframes"""
import dgl

__all__ = ['PandasGraphBuilder']


class PandasGraphBuilder(object):
    """
    Creates a heterogeneous graph from multiple pandas dataframes.
    """

    def __init__(self):
        self.entity_pk_to_name = {}  # mapping from primary key name to entity name
        self.num_nodes_per_type = {}
        self.edges_per_relation = {}

    def add_entities(self, entity_table, primary_key, name):
        self.entity_pk_to_name[primary_key] = name
        self.num_nodes_per_type[name] = entity_table.shape[0]

    def add_binary_relations(self, relation_table, source_key, destination_key, name):
        srctype = self.entity_pk_to_name[source_key]
        dsttype = self.entity_pk_to_name[destination_key]
        etype = (srctype, name, dsttype)
        self.edges_per_relation[etype] = (
            relation_table[source_key].values.astype('int64'), relation_table[destination_key].values.astype('int64'))

    def build(self):
        graph = dgl.heterograph(self.edges_per_relation, self.num_nodes_per_type)
        return graph
