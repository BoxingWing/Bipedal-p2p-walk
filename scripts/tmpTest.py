# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# import torch
# import unittest
# from collections import deque

# """Launch Isaac Sim Simulator first."""

# from isaaclab.app import AppLauncher, run_tests

# # launch omniverse app in headless mode
# simulation_app = AppLauncher(headless=True).app

# """Rest everything follows from here."""

# from isaaclab.utils import CircularBuffer


# class TestCircularBuffer(unittest.TestCase):
#     """Test fixture for checking the circular buffer implementation."""

#     def setUp(self):
#         self.max_len = 5
#         self.batch_size = 3
#         self.device = "cpu"
#         self.buffer = CircularBuffer(self.max_len, self.batch_size, self.device)

#     """
#     Test cases for CircularBuffer class.
#     """

#     def test_initialization(self):
#         """Test initialization of the circular buffer."""
#         self.assertEqual(self.buffer.max_length, self.max_len)
#         self.assertEqual(self.buffer.batch_size, self.batch_size)
#         self.assertEqual(self.buffer.device, self.device)
#         self.assertEqual(self.buffer.current_length.tolist(), [0, 0, 0])

#     def test_reset(self):
#         """Test resetting the circular buffer."""
#         # append some data
#         data = torch.ones((self.batch_size, 2), device=self.device)
#         self.buffer.append(data)
#         # reset the buffer
#         self.buffer.reset()

#         # check if the buffer has zeros entries
#         self.assertEqual(self.buffer.current_length.tolist(), [0, 0, 0])

#     def test_reset_subset(self):
#         """Test resetting a subset of batches in the circular buffer."""
#         data1 = torch.ones((self.batch_size, 2), device=self.device)
#         data2 = 2.0 * data1.clone()
#         data3 = 3.0 * data1.clone()
#         self.buffer.append(data1)
#         self.buffer.append(data2)
#         # reset the buffer
#         reset_batch_id = 1
#         self.buffer.reset(batch_ids=[reset_batch_id])
#         # check that correct batch is reset
#         self.assertEqual(self.buffer.current_length.tolist()[reset_batch_id], 0)
#         # Append new set of data
#         self.buffer.append(data3)
#         # check if the correct number of entries are in each batch
#         expected_length = [3, 3, 3]
#         expected_length[reset_batch_id] = 1
#         self.assertEqual(self.buffer.current_length.tolist(), expected_length)
#         # check that all entries of the recently reset and appended batch are equal
#         for i in range(self.max_len):
#             torch.testing.assert_close(self.buffer.buffer[reset_batch_id, 0], self.buffer.buffer[reset_batch_id, i])

#     def test_append_and_retrieve(self):
#         """Test appending and retrieving data from the circular buffer."""
#         # append some data
#         data1 = torch.tensor([[1, 1], [1, 1], [1, 1]], device=self.device)
#         data2 = torch.tensor([[2, 2], [2, 2], [2, 2]], device=self.device)

#         self.buffer.append(data1)
#         self.buffer.append(data2)

#         self.assertEqual(self.buffer.current_length.tolist(), [2, 2, 2])

#         retrieved_data = self.buffer[torch.tensor([0, 0, 0], device=self.device)]
#         self.assertTrue(torch.equal(retrieved_data, data2))

#         retrieved_data = self.buffer[torch.tensor([1, 1, 1], device=self.device)]
#         self.assertTrue(torch.equal(retrieved_data, data1))

#     def test_buffer_overflow(self):
#         """Test buffer overflow.

#         If the buffer is full, the oldest data should be overwritten.
#         """
#         # add data in ascending order
#         for count in range(self.max_len + 2):
#             data = torch.full((self.batch_size, 4), count, device=self.device)
#             self.buffer.append(data)

#         # check buffer length is correct
#         self.assertEqual(self.buffer.current_length.tolist(), [self.max_len, self.max_len, self.max_len])

#         # retrieve most recent data
#         key = torch.tensor([0, 0, 0], device=self.device)
#         retrieved_data = self.buffer[key]
#         expected_data = torch.full_like(data, self.max_len + 1)

#         self.assertTrue(torch.equal(retrieved_data, expected_data))

#         # retrieve the oldest data
#         key = torch.tensor([self.max_len - 1, self.max_len - 1, self.max_len - 1], device=self.device)
#         retrieved_data = self.buffer[key]
#         expected_data = torch.full_like(data, 2)

#         self.assertTrue(torch.equal(retrieved_data, expected_data))

#     def test_empty_buffer_access(self):
#         """Test accessing an empty buffer."""
#         with self.assertRaises(RuntimeError):
#             self.buffer[torch.tensor([0, 0, 0], device=self.device)]

#     def test_invalid_batch_size(self):
#         """Test appending data with an invalid batch size."""
#         data = torch.ones((self.batch_size + 1, 2), device=self.device)
#         with self.assertRaises(ValueError):
#             self.buffer.append(data)

#         with self.assertRaises(ValueError):
#             self.buffer[torch.tensor([0, 0], device=self.device)]

#     def test_key_greater_than_pushes(self):
#         """Test retrieving data with a key greater than the number of pushes.

#         In this case, the oldest data should be returned.
#         """
#         data1 = torch.tensor([[1, 1], [1, 1], [1, 1]], device=self.device)
#         data2 = torch.tensor([[2, 2], [2, 2], [2, 2]], device=self.device)

#         self.buffer.append(data1)
#         self.buffer.append(data2)

#         retrieved_data = self.buffer[torch.tensor([5, 5, 5], device=self.device)]
#         self.assertTrue(torch.equal(retrieved_data, data1))

#     def test_return_buffer_prop(self):
#         """Test retrieving the whole buffer for correct size and contents.
#         Returning the whole buffer should have the shape [batch_size,max_len,data.shape[1:]]
#         """
#         num_overflow = 2
#         for i in range(self.buffer.max_length + num_overflow):
#             data = torch.tensor([[i]], device=self.device).repeat(3, 2)
#             self.buffer.append(data)

#         retrieved_buffer = self.buffer.buffer
#         # check shape
#         self.assertTrue(retrieved_buffer.shape == torch.Size([self.buffer.batch_size, self.buffer.max_length, 2]))
#         # check that batch is first dimension
#         torch.testing.assert_close(retrieved_buffer[0], retrieved_buffer[1])
#         # check oldest
#         torch.testing.assert_close(
#             retrieved_buffer[:, 0], torch.tensor([[num_overflow]], device=self.device).repeat(3, 2)
#         )
#         # check most recent
#         torch.testing.assert_close(
#             retrieved_buffer[:, -1],
#             torch.tensor([[self.buffer.max_length + num_overflow - 1]], device=self.device).repeat(3, 2),
#         )
#         # check that it is returned oldest first
#         for idx in range(self.buffer.max_length - 1):
#             self.assertTrue(torch.all(torch.le(retrieved_buffer[:, idx], retrieved_buffer[:, idx + 1])))


# def main():
#     a = 0.45
#     tensor = torch.tensor([1.5, 2.7, 4.2, 5.1])  # 这里是一个示例张量

#     # 变成 a 的整数倍
#     result = torch.round(tensor / a) * a
#     print(result)
#     # """Main function."""
#     # max_len = 5
#     # num_envs = 3
#     # action_num = 3
#     # batch_size = num_envs
#     # device = "cuda"
#     # buffer = CircularBuffer(max_len, batch_size, device)
#     # data1 = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], device=device)
#     # data2 = torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=device)
#     # data3 = torch.tensor([[3, 3, 3], [3, 3, 3], [3, 3, 3]], device=device)
#     # data4 = torch.tensor([[4, 4, 4], [4, 4, 4], [4, 4, 4]], device=device)
#     # data5 = torch.tensor([[5, 5, 5], [5, 5, 5], [5, 5, 5]], device=device)
#     # buffer.append(data1)
#     # buffer.append(data2)
#     # buffer.append(data3)
#     # buffer.append(data4)
#     # buffer.append(data5)
    
#     # # print(buffer[torch.full((batch_size,), 0, device=device)])
#     # # print(buffer.buffer[:,0,:])
#     # print(buffer.buffer[:,0,:])
#     # print(buffer.buffer[:,1,:])
#     # print(buffer.buffer[:,2,:])
#     # print(buffer.buffer[:,3,:])
#     # print(buffer.buffer[:,4,:])
    

#     # env_ids = [0,2]
#     # # for i in range(max_len):
#     # #     buffer.buffer[:,i,:][env_ids]*=0
#     # #     # buffer[torch.full((batch_size,), i, device=device)][env_ids,:] *= 0
#     # #     # print(buffer[torch.full((batch_size,), i, device=device)])
#     # for i in range(max_len):
#     #     buffer._buffer[i, env_ids, : ] = 0  # Clear specific rows

#     # print("ssssssssssss")
#     # print(buffer.buffer[:,0,:])
#     # print(buffer.buffer[:,1,:])
#     # print(buffer.buffer[:,2,:])
#     # print(buffer.buffer[:,3,:])
#     # print(buffer.buffer[:,4,:])
    

#     # print("=======")
#     # buffer_deque = deque(maxlen=max_len)
#     # buffer_deque.append(data1)
#     # buffer_deque.append(data2)
#     # buffer_deque.append(data3)
#     # print(buffer_deque[0])
#     # print(buffer_deque[1])
#     # print(buffer_deque[2])

# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app

class cmd_scheduler:
    def __init__(self, run_dt, gait_cycle_time):
        self._start_count = []
        self._period = []
        self._px = []
        self._py = []
        self._thetaZ = []
        self._delta_phi = []
        self._delta_phi_gait = []
        self._run_dt = run_dt
        self._gait_cycle_time = gait_cycle_time

        self.phi = 0.0
        self.phi_gait = 0.0
        self._phi_pseudo = 0.0
        self._current_idx = 0

        self.cmd_cur = [0.0, 0.0, 0.0]
        self.cmd_next = [0.0, 0.0, 0.0]
    
    def add_mode(self, period, px, py, thetaZ, isStand):
        # self._start_count.append(start_count)
        self._period.append(period)
        self._px.append(px)
        self._py.append(py)
        self._thetaZ.append(thetaZ)
        if not(isStand):
            self._delta_phi.append(self._run_dt / period)
            self._delta_phi_gait.append(self._run_dt / self._gait_cycle_time)
        else:
            self._delta_phi.append(0.0)
            self._delta_phi_gait.append(0.0)
    
    def step(self):

        if self._current_idx >= len(self._px):
            self._phi_pseudo = 0.0
            self.phi = 0.0
            self.phi_gait = 0.0

            self.cmd_cur[0] = 0.0
            self.cmd_cur[1] = 0.0
            self.cmd_cur[2] = 0.0
        else:
            self._phi_pseudo += self._run_dt / self._period[self._current_idx]
            self.phi += self._delta_phi[self._current_idx]
            self.phi_gait += self._delta_phi_gait[self._current_idx]

            self.cmd_cur[0] = self._px[self._current_idx]
            self.cmd_cur[1] = self._py[self._current_idx]
            self.cmd_cur[2] = self._thetaZ[self._current_idx]

        if self._current_idx+1 >= len(self._px):
            self.cmd_next[0] = 0.0
            self.cmd_next[1] = 0.0
            self.cmd_next[2] = 0.0
        else:
            self.cmd_next[0] = self._px[self._current_idx+1]
            self.cmd_next[1] = self._py[self._current_idx+1]
            self.cmd_next[2] = self._thetaZ[self._current_idx+1]
        
        if self._phi_pseudo>=1:
            self._current_idx += 1
            self._phi_pseudo = 0.0
            self.phi = 0.0
            self.phi_gait = 0.0

        
def main():
    cmd = cmd_scheduler(run_dt=0.01,gait_cycle_time=0.9)
    cmd.add_mode(period=2.,px=0.2,py=0.2,thetaZ=1.5708,isStand=False)
    cmd.add_mode(period=1.,px=0.0,py=0.0,thetaZ=0.0,isStand=True)
    cmd.add_mode(period=3.,px=0.3,py=0.3,thetaZ=0.7,isStand=False)

    for i in range(1000):
        cmd.step()
        print("===========")
        print(i)
        print(cmd.phi)
        print(cmd.phi_gait)
        print(cmd.cmd_cur)
        print(cmd.cmd_next)

if __name__ == "__main__":
    main()















        









