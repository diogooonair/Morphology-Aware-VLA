import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import torch


KINEMATIC_INPUT_DIM = 12


def parse_urdf_to_morphology(urdf_path: str, max_joints: int = 18) -> dict:
    """
    Parse a URDF into a topologically ordered morphology representation.

    Per-joint feature vector (12D):
        [lower_limit, upper_limit,
         axis_x, axis_y, axis_z,
         joint_type,
         origin_x, origin_y, origin_z,
         origin_roll, origin_pitch, origin_yaw]
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    active_types = {"revolute", "continuous", "prismatic"}
    link_tree = nx.DiGraph()
    joint_dict = {}

    for joint_elem in root.findall("joint"):
        joint_name = joint_elem.get("name")
        joint_type = joint_elem.get("type")

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue

        parent_link = parent_elem.get("link")
        child_link = child_elem.get("link")

        if joint_type not in active_types:
            continue

        # Axis
        axis_elem = joint_elem.find("axis")
        if axis_elem is not None:
            axis = [float(v) for v in axis_elem.get("xyz", "0 0 1").split()]
        else:
            axis = [0.0, 0.0, 1.0]

        # Limits
        limit_elem = joint_elem.find("limit")
        if joint_type == "continuous":
            lower, upper = -np.pi, np.pi
        elif limit_elem is not None:
            lower = float(limit_elem.get("lower", -np.pi))
            upper = float(limit_elem.get("upper", np.pi))
        else:
            lower, upper = -np.pi, np.pi

        # Type encoding
        type_val = 1.0 if joint_type in ("revolute", "continuous") else 2.0

        # Origin
        origin_elem = joint_elem.find("origin")
        if origin_elem is not None:
            origin_xyz = [float(v) for v in origin_elem.get("xyz", "0 0 0").split()]
            origin_rpy = [float(v) for v in origin_elem.get("rpy", "0 0 0").split()]
        else:
            origin_xyz = [0.0, 0.0, 0.0]
            origin_rpy = [0.0, 0.0, 0.0]

        features = [
            lower, upper,
            axis[0], axis[1], axis[2],
            type_val,
            origin_xyz[0], origin_xyz[1], origin_xyz[2],
            origin_rpy[0], origin_rpy[1], origin_rpy[2],
        ]

        joint_dict[joint_name] = {
            "name": joint_name,
            "parent_link": parent_link,
            "child_link": child_link,
            "features": features,
        }
        link_tree.add_edge(parent_link, child_link, joint_name=joint_name)

    if not joint_dict:
        raise ValueError(f"No active joints found in URDF: {urdf_path}")

    roots = [n for n, d in link_tree.in_degree() if d == 0]
    if not roots:
        roots = [list(joint_dict.values())[0]["parent_link"]]

    ordered_joint_names = []
    for edge in nx.bfs_edges(link_tree, source=roots[0]):
        j_name = link_tree.edges[edge].get("joint_name")
        if j_name in joint_dict and j_name not in ordered_joint_names:
            ordered_joint_names.append(j_name)

    active_joints = [joint_dict[n] for n in ordered_joint_names][:max_joints]

    # Padded tensors
    joint_features = np.zeros((max_joints, KINEMATIC_INPUT_DIM), dtype=np.float32)
    dof_mask = np.zeros(max_joints, dtype=np.bool_)

    for i, joint in enumerate(active_joints):
        joint_features[i] = joint["features"]
        dof_mask[i] = True

    return {
        "joint_features": torch.tensor(joint_features, dtype=torch.float32),
        "dof_mask": torch.tensor(dof_mask, dtype=torch.bool),
    }


class MorphologyCache:
    def __init__(self, max_joints: int = 18):
        self.max_joints = max_joints
        self._cache = {}

    def get_morphology(self, robot_id: str, urdf_path: str, device: torch.device) -> dict:
        if robot_id not in self._cache:
            self._cache[robot_id] = parse_urdf_to_morphology(urdf_path, self.max_joints)
        cached = self._cache[robot_id]
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cached.items()}