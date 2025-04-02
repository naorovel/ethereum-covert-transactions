from json import loads, dumps
import pandas as pd
import networkx as nx
import random
import itertools
import json

num_transactions_display = 100000

###### Creating graph
transactions_df = pd.read_csv("./src/data/transactions.csv")

display_transactions=transactions_df.head(num_transactions_display)

transactions_df = display_transactions

unique_addr = transactions_df['from_address'].unique().tolist() + transactions_df['to_address'].unique().tolist()

nodes = []

for node in unique_addr: 
    nodes.append({'id': node})
        
    transactions_df["link"] = transactions_df.apply(lambda x: {'source': x["from_address"], 
                                                                'target': x["to_address"],
                                                                'block_timestamp': x["block_timestamp"],
                                                                'transaction_index': x["transaction_index"],
                                                                'hash': x['hash'],
                                                                'value': x['value']}, axis=1)
        
    links = transactions_df["link"].values.tolist()
    

def detect_covert_subgraphs(G):
    # Calculate structural features for all subgraphs
    components = (nx.weakly_connected_components(G) if G.is_directed() 
                 else nx.connected_components(G))
    
    all_features = []
    subgraph_features = {}
    
    # First pass: Calculate features for all subgraphs
    for i, component in enumerate(components):
        sg = G.subgraph(component)
        undir_sg = sg.to_undirected(as_view=True)
        
        # Calculate subgraph features with multigraph support
        features = {
            'size': len(component),
            'diameter': nx.diameter(undir_sg) if len(component) > 1 else 0,
            'avg_degree': sum(dict(sg.degree()).values()) / len(component) if len(component) > 0 else 0,
            'max_in_degree': max(dict(sg.in_degree()).values()) if G.is_directed() else 0,
            'max_out_degree': max(dict(sg.out_degree()).values()) if G.is_directed() else 0
        }
        
        # Handle multigraph-specific features
        if G.is_multigraph():
            features['edge_multiplicity'] = sum(
                len(data) for u, v, data in sg.edges(data=True)
            ) / sg.number_of_edges() if sg.number_of_edges() > 0 else 0
        
        subgraph_features[i] = features
        all_features.extend(features.values())
    
    # Calculate threshold (Th) and epsilon (ε)
    Th = random.choice(all_features)
    pairs = itertools.combinations(all_features, 2)
    eps = min(abs(a-b) for a, b in pairs if a != b) if len(all_features) > 1 else 0
    
    # Second pass: Detect covert subgraphs
    components = (nx.weakly_connected_components(G) if G.is_directed() 
                 else nx.connected_components(G))
    
    covert_subgraphs = []
    for i, component in enumerate(components):
        features = subgraph_features[i]
        is_covert = any(abs(Th - val) <= eps for val in features.values())
        
        if is_covert:
            covert_subgraphs.append(component)
            # Label nodes
            for node in component:
                G.nodes[node]['covert'] = True
            # Label edges with proper multigraph handling
            for edge in G.subgraph(component).edges(keys=G.is_multigraph()):
                if G.is_multigraph():
                    u, v, k = edge
                    G.edges[u, v, k]['covert'] = True
                else:
                    u, v = edge
                    G.edges[u, v]['covert'] = True
    
    return G, {
        'threshold': Th,
        'epsilon': eps,
        'covert_subgraphs': covert_subgraphs
    }

def count_covert_transactions(G):
    """Count covert and non-covert edges in any graph type"""
    covert = 0
    non_covert = 0
    
    # Handle all edge types (MultiGraph and regular)
    for edge in G.edges(keys=True, data=True):
        # edge format:
        # - (u, v, data) for regular graphs
        # - (u, v, key, data) for multigraphs
        edge_data = edge[-1]  # Always get data dict
        if edge_data.get('covert', False):
            covert += 1
        else:
            non_covert += 1
            
    return covert, non_covert

# Load graph with proper multigraph handling
G = nx.node_link_graph({'directed': True, 'multigraph': True, 'graph': {}, 'nodes': nodes, 'links': links})    

# Process the graph
annotated_graph, stats = detect_covert_subgraphs(G)

# Display results
print(f"Detection Threshold (Th): {stats['threshold']}")
print(f"Base Offset (ε): {stats['epsilon']}")
print("\nNode Attributes:")
print(annotated_graph.nodes(data=True))
print("\nEdge Attributes:")
print(list(annotated_graph.edges(keys=True, data=True)))

# Count transactions
covert, normal = count_covert_transactions(annotated_graph)

# Show detailed results
print("Edge Type Breakdown:")
for edge in annotated_graph.edges(keys=True, data=True):
    edge_type = "COVERT" if edge[-1].get('covert') else "Normal"
    print(f"- {edge[:3]}: {edge_type}")

print(f"\nFinal Counts: {covert} covert, {normal} normal")
