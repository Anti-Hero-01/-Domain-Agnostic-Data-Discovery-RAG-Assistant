# backend/graph.py
import networkx as nx
import json
from pathlib import Path
GRAPH_FILE = Path('./graph.json')

G = nx.MultiDiGraph()
if GRAPH_FILE.exists():
    try:
        obj = json.loads(GRAPH_FILE.read_text())
        G = nx.node_link_graph(obj)
    except Exception:
        G = nx.MultiDiGraph()

def add_triples_to_graph(doc_id, triple_payload):
    # triple_payload = {'entities': [...], 'triples': [{'subj':..., 'pred':..., 'obj':...}, ...]}
    for t in triple_payload.get('triples', []):
        subj = t.get('subj')
        pred = t.get('pred')
        obj = t.get('obj')
        if not (subj and pred and obj):
            continue
        G.add_node(subj, type='entity')
        G.add_node(obj, type='entity')
        G.add_edge(subj, obj, key=pred, relation=pred, source_doc=doc_id)
    persist_graph()

def persist_graph():
    data = nx.node_link_data(G)
    GRAPH_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def graph_summary():
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges()
    }
