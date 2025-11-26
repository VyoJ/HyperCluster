"""
Network topology management for distributed inference.
Adapted from exo's topology system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from device_capabilities import DeviceCapabilities


@dataclass
class PeerConnection:
    """Represents a connection between two peers."""

    from_id: str
    to_id: str
    description: Optional[str] = None

    def __hash__(self):
        return hash((self.from_id, self.to_id))

    def __eq__(self, other):
        if not isinstance(other, PeerConnection):
            return False
        return self.from_id == other.from_id and self.to_id == other.to_id


class Topology:
    """
    Manages the network topology of HyperCluster nodes.

    Tracks:
    - Node capabilities (memory, compute)
    - Peer connections (who can talk to whom)
    - Active node for inference coordination
    """

    def __init__(self):
        self.nodes: Dict[str, DeviceCapabilities] = {}
        self.peer_graph: Dict[str, Set[PeerConnection]] = {}
        self.active_node_id: Optional[str] = None

    def update_node(self, node_id: str, device_capabilities: DeviceCapabilities):
        """Update or add a node's capabilities."""
        self.nodes[node_id] = device_capabilities

    def get_node(self, node_id: str) -> Optional[DeviceCapabilities]:
        """Get a node's capabilities."""
        return self.nodes.get(node_id)

    def all_nodes(self) -> List[Tuple[str, DeviceCapabilities]]:
        """Get all nodes as list of (node_id, capabilities) tuples."""
        return list(self.nodes.items())

    def add_edge(self, from_id: str, to_id: str, description: Optional[str] = None):
        """Add a connection between two nodes."""
        if from_id not in self.peer_graph:
            self.peer_graph[from_id] = set()
        conn = PeerConnection(from_id, to_id, description)
        self.peer_graph[from_id].add(conn)

    def remove_node(self, node_id: str):
        """Remove a node from the topology."""
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.peer_graph:
            del self.peer_graph[node_id]
        # Remove edges pointing to this node
        for connections in self.peer_graph.values():
            connections_to_remove = [c for c in connections if c.to_id == node_id]
            for conn in connections_to_remove:
                connections.remove(conn)

    def merge(self, peer_node_id: str, other: "Topology"):
        """
        Merge another topology into this one.
        Used when receiving topology info from peers.
        """
        for node_id, capabilities in other.nodes.items():
            if node_id != peer_node_id:
                continue
            self.update_node(node_id, capabilities)

        for node_id, connections in other.peer_graph.items():
            for conn in connections:
                if conn.from_id != peer_node_id:
                    continue
                self.add_edge(conn.from_id, conn.to_id, conn.description)

    def is_fully_connected(self) -> bool:
        """Check if all nodes can reach each other (directly or indirectly)."""
        if len(self.nodes) <= 1:
            return True

        # Simple reachability check using BFS
        start_node = next(iter(self.nodes.keys()))
        visited = set()
        queue = [start_node]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Add neighbors
            if current in self.peer_graph:
                for conn in self.peer_graph[current]:
                    if conn.to_id not in visited:
                        queue.append(conn.to_id)

        return len(visited) == len(self.nodes)

    def __str__(self):
        nodes_str = ", ".join(
            f"{node_id}: {cap}" for node_id, cap in self.nodes.items()
        )
        edges_str = ", ".join(
            f"{node}: {[f'{c.to_id}({c.description})' for c in conns]}"
            for node, conns in self.peer_graph.items()
        )
        return f"Topology(Nodes: {{{nodes_str}}}, Edges: {{{edges_str}}})"

    def to_json(self) -> dict:
        """Convert topology to JSON-serializable dict."""
        return {
            "nodes": {
                node_id: capabilities.to_dict()
                for node_id, capabilities in self.nodes.items()
            },
            "peer_graph": {
                node_id: [
                    {
                        "from_id": conn.from_id,
                        "to_id": conn.to_id,
                        "description": conn.description,
                    }
                    for conn in connections
                ]
                for node_id, connections in self.peer_graph.items()
            },
            "active_node_id": self.active_node_id,
        }

    @staticmethod
    def from_json(data: dict) -> "Topology":
        """Create topology from JSON dict."""
        topology = Topology()

        # Restore nodes
        for node_id, cap_dict in data.get("nodes", {}).items():
            topology.update_node(node_id, DeviceCapabilities.from_dict(cap_dict))

        # Restore edges
        for node_id, connections in data.get("peer_graph", {}).items():
            for conn_dict in connections:
                topology.add_edge(
                    conn_dict["from_id"],
                    conn_dict["to_id"],
                    conn_dict.get("description"),
                )

        # Restore active node
        topology.active_node_id = data.get("active_node_id")

        return topology
