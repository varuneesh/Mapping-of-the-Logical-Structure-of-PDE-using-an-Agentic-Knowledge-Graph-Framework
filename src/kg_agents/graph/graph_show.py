import json
from pyvis.network import Network
from pathlib import Path
import webbrowser

# =========================
# 1. Load JSON file
# =========================
ROOT_DIR = Path.cwd().parent.parent.parent
GRAPH      = ROOT_DIR / "data" / "graph_memory.json"

with open(GRAPH, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# 2. Create network
# =========================
net = Network(
    height="800px",
    width="100%",
    directed=True,
    bgcolor="#111111",
    font_color="white"
)

# Enable physics for better layout
net.barnes_hut()

# =========================
# 3. Add nodes
# =========================
def get_color(node_type):
    if node_type == "NumericalMethod":
        return "#1f77b4"  # blue
    elif node_type == "ErrorConcept":
        return "#d62728"  # red
    elif node_type == "Theorem":
        return "#2ca02c"  # green
    elif node_type == "Definition":
        return "#ff7f0e"  # orange
    elif node_type == "ProblemType":
        return "#9467bd"  # purple
    else:
        return "#7f7f7f"  # gray

for node, attrs in data["nodes"].items():
    node_type = attrs.get("type", "Unknown")

    net.add_node(
        node,
        label=node,
        title=f"""
        <b>{node}</b><br>
        Type: {node_type}<br>
        Salience: {attrs.get('salience', 'N/A')}<br>
        Sources: {len(attrs.get('sources', []))}
        """,
        color=get_color(node_type),
        size=10 + attrs.get("salience", 0) * 5  # scale size
    )

# =========================
# 4. Add edges (triples)
# =========================
for edge in data["edges"]:
    src = edge["source"]
    tgt = edge["target"]
    rel = edge.get("relation", "")

    net.add_edge(
        src,
        tgt,
        label=rel,
        title=rel
    )

# =========================
# 5. Improve layout settings
# =========================
net.set_options("""
var options = {
  "nodes": {
    "font": {
      "size": 14
    }
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true
      }
    },
    "font": {
      "size": 10,
      "align": "middle"
    },
    "smooth": false
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -30000,
      "centralGravity": 0.3,
      "springLength": 150,
      "springConstant": 0.04
    },
    "minVelocity": 0.75
  }
}
""")

# =========================
# 6. Save and open
# =========================
net.write_html("knowledge_graph.html")
webbrowser.open("knowledge_graph.html")