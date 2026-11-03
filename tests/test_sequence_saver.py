"""Checks the rollout frame schema (README, Simulation) without a CtRL-Sim checkpoint.

Run from the repo root: python tests/test_sequence_saver.py   (pytest also works)
"""
import os
import pickle
import sys
import tempfile

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import Simulator
from utils.sequence_saver import SequenceSaver


def _scene():
    # two agents + ego (last row): [x, y, vx, vy, theta, l, w, exists, z, height]
    agents = np.array([
        [10., 0., 5., 0., 0., 4.5, 2., 1., 0.1, 1.6],
        [0., 5., 0., 0., np.pi / 2, 0.8, 0.8, 1., 0., 1.7],
        [0., 0., 8., 0., 0., 5., 2.2, 1., 0., 1.8],
    ])[:, None]
    lanes = np.stack([np.stack([np.linspace(-50, 50, 50), np.full(50, y)], -1) for y in (0., 3.5)])
    cam_infos = {'CAM_F0': {'ego_pose': [101.5, 200., 1.5, 1., 0., 0., 0.], 'filename_jpg': 'x.jpg'}}
    ego_state_og = [np.array([100., 200., 0.]), np.array([1., 0., 0., 0.]), np.array([5., 2.2, 1.8]), 0.]
    return {
        'agents': agents, 'agent_types': np.eye(5)[[1, 2, 1]], 'lanes': lanes, 'lane_graph': nx.DiGraph([(0, 1)]),
        'num_lanes': 2, 'route': np.stack([np.arange(50.), np.zeros(50)], -1), 'route_lane_indices': [0],
        'cam_infos': cam_infos, 'ego_state_og': ego_state_og,
    }


def _sim_at(scene, ego_state):
    """ Simulator in the state get_serializable_state() reads, without loading a model."""
    sim = Simulator.__new__(Simulator)
    sim.ego_trajectory = scene['agents'][-1]
    sim.ego_state = ego_state
    sim.scenario_dict = {'lanes': scene['lanes']}
    sim.data_dict = {'agent': [scene['agents'][:-1, 0]], 'agent_type': [scene['agent_types'][:-1]]}
    sim.agent_active = np.array([True, False])
    sim.cam_infos = scene['cam_infos']
    sim.ego_state_og_initial = scene['ego_state_og']
    return sim


def test_frame_schema():
    scene = _scene()
    # ego moved 10 m forward and turned 90 deg left
    ego_t = np.array([10., 0., 0., 8., np.pi / 2, 5., 2.2, 1., 0., 1.8])
    with tempfile.TemporaryDirectory() as out:
        saver = SequenceSaver(out, 'scene_seed0', scene)
        saver.save_frame(0, _sim_at(scene, scene['agents'][-1, 0].copy()).get_serializable_state())
        saver.save_frame(1, _sim_at(scene, ego_t).get_serializable_state())
        assert sorted(os.listdir(os.path.join(out, 'scene_seed0'))) == ['frame_0000.pkl', 'frame_0001.pkl']
        with open(os.path.join(out, 'scene_seed0', 'frame_0001.pkl'), 'rb') as f:
            frame = pickle.load(f)

    assert {'agents', 'agent_types', 'agent_active', 'lanes', 'lane_graph', 'route', 'route_lane_indices',
            'num_agents', 'num_lanes', 'cam_infos', 'ego_state_og'} <= set(frame)
    assert frame['agents'].shape == (3, 1, 10) and frame['agents'].dtype == np.float64
    assert frame['agent_types'].shape == (3, 5) and frame['num_agents'] == 3
    assert frame['agent_active'].tolist() == [True, False, True]  # ego appended, always active
    assert frame['lanes'].shape == (2, 50, 2)

    # ego is the last row, at the origin facing +x, keeping length/width/height
    np.testing.assert_allclose(frame['agents'][-1, 0], [0, 0, 0, 0, 0, 5., 2.2, 1., 0., 1.8])
    assert frame['agent_types'][-1].tolist() == [0, 1, 0, 0, 0]
    # agent 1 at start-frame (0, 5) is 5 m ahead, 10 m left of the turned ego (x forward, y left)
    np.testing.assert_allclose(frame['agents'][1, 0, :2], [5., 10.], atol=1e-9)
    np.testing.assert_allclose(frame['agents'][1, 0, 4], 0., atol=1e-9)  # heading +y in start frame = ego heading
    np.testing.assert_allclose(frame['agents'][0, 0, 2:4], [0., -5.], atol=1e-9)  # velocity rotated into ego frame
    np.testing.assert_allclose(frame['agents'][0, 0, 8:10], [0.1, 1.6])  # z, height carried through

    # global ego pose follows the ego: start heading 0 at (100, 200) -> (110, 200), heading +90 deg
    translation, quat, dims, heading = frame['ego_state_og']
    np.testing.assert_allclose(translation, [110., 200., 0.], atol=1e-9)
    np.testing.assert_allclose(heading, np.pi / 2)
    np.testing.assert_allclose(quat, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], atol=1e-9)
    np.testing.assert_allclose(dims, [5., 2.2, 1.8])
    # camera keeps its 1.5 m forward mount offset, rotated with the ego
    np.testing.assert_allclose(frame['cam_infos']['CAM_F0']['ego_pose'][:3], [110., 201.5, 1.5], atol=1e-9)
    assert frame['cam_infos']['CAM_F0']['filename_jpg'] == 'x.jpg'


if __name__ == '__main__':
    test_frame_schema()
    print('ok')
