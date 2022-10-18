
# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import matplotlib.pyplot as plt

from tests.lava.test_utils.utils import Utils

from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.runtime.runtime import Runtime

from lava.magma.core.run_conditions import RunSteps

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from sparse import Sparse

from lava.lib.dnf.connect.connect import connect, _configure_ops, \
    _compute_weights
from lava.lib.dnf.kernels.kernels import MultiPeakKernel
from lava.lib.dnf.operations.operations import Convolution

from dvs.file.dvs_file_input.process import DVSFileInput, PyDVSFileInputPMDense
from miscellaneous.c_injector.process import CInjector, CInjectorPMVecDense
from miscellaneous.c_spike_reader.process import CSpikeReader, \
    CSpikeReaderPMVecDense
from miscellaneous.py_spike_reader.process import PySpikeReader, PySpikeReaderPMDense


class TestOasis(unittest.TestCase):
    run_it_tests: bool = Utils.get_bool_env_setting("RUN_IT_TESTS")

    # Architecture:
    # DVSFileInput -> CInjector -> Dense -> LIF -> CSpikeReader -> PySpikeReader
    @unittest.skipUnless(run_it_tests, "")
    def test_basic_run_and_stop_dense_ports_dense_synapses(self):
        # Run Params
        exception_pm_map = {
            DVSFileInput: PyDVSFileInputPMDense,
            CInjector: CInjectorPMVecDense,
            CSpikeReader: CSpikeReaderPMVecDense,
            PySpikeReader: PySpikeReaderPMDense
        }
        num_steps = 200
        run_cfg = Loihi2HwCfg(exception_proc_model_map=exception_pm_map)
        run_cnd = RunSteps(num_steps=num_steps)

        # Process Params
        true_height = 180
        true_width = 240
        file_path = "twoHands-2022_07_19_11_16_19.aedat4"
        flatten = True
        down_sample_factor = 8
        down_sample_mode = "max_pooling"  # down_sample, convolution

        down_sampled_shape = (true_height // down_sample_factor,
                              true_width // down_sample_factor)

        num_neurons = (true_height // down_sample_factor) * \
                      (true_width // down_sample_factor)
        down_sampled_flat_shape = (num_neurons,)

        # DNF Params
        kernel = MultiPeakKernel(amp_exc=10,
                                 width_exc=[7, 7],
                                 amp_inh=-5,
                                 width_inh=[14, 14])

        # Define Processes
        dvs_file_input = DVSFileInput(true_height=true_height,
                                      true_width=true_width,
                                      file_path=file_path,
                                      flatten=flatten,
                                      down_sample_factor=down_sample_factor,
                                      down_sample_mode=down_sample_mode)
        # replace with PyToNxAdapter?
        c_injector = CInjector(shape=down_sampled_flat_shape)
        dense = Sparse(weights=np.eye(num_neurons) * 10)
        lif = LIF(shape=down_sampled_shape, du=4095, dv=4095, vth=5)
        c_spike_reader = CSpikeReader(shape=down_sampled_shape)
        py_spike_reader = PySpikeReader(shape=down_sampled_shape)

        # Connect Processes
        dvs_file_input.event_frame_out.connect(c_injector.in_port)
        c_injector.out_port.connect(dense.s_in)
        dense.a_out.reshape(new_shape=down_sampled_shape).connect(lif.a_in)
        lif.s_out.connect(c_spike_reader.in_port)
        c_spike_reader.out_port.connect(py_spike_reader.in_port)

        # DNF connection
        ops = [Convolution(kernel)]
        _configure_ops(ops, lif.s_out.shape, lif.a_in.shape)
        weights = _compute_weights(ops)
        connections = Sparse(weights=weights)
        con_ip = connections.s_in
        lif.s_out.reshape(new_shape=con_ip.shape).connect(con_ip)
        con_op = connections.a_out
        con_op.reshape(new_shape=lif.a_in.shape).connect(lif.a_in)

        print(100 * "=")

        # Compilation
        print("Compiling : BEGIN")
        compiler = Compiler()
        executable = compiler.compile(dvs_file_input, run_cfg=run_cfg)
        print("Compiling : END")

        print(100 * "=")

        # Initializing runtime
        print("Initializing Runtime : BEGIN")
        mp = ActorType.MultiProcessing
        runtime = Runtime(exe=executable,
                          message_infrastructure_type=mp)
        runtime.initialize()
        print("Initializing Runtime : END")

        print(100 * "=")

        # Running for num_steps time steps
        print("Running : BEGIN")
        runtime.start(run_condition=run_cnd)
        runtime.pause()

        dnf_spikes_0 = py_spike_reader.spikes.get()

        runtime.start(run_condition=run_cnd)
        runtime.pause()

        dnf_spikes_1 = py_spike_reader.spikes.get()

        runtime.start(run_condition=run_cnd)
        runtime.pause()

        dnf_spikes_2 = py_spike_reader.spikes.get()

        print("Running : END")

        print(100 * "=")

        print("Stopping : BEGIN")
        runtime.stop()
        print("Stopping : END")

        fig = plt.figure()
        plt.imshow(dnf_spikes_0)
        fig = plt.figure()
        plt.imshow(dnf_spikes_1)
        fig = plt.figure()
        plt.imshow(dnf_spikes_2)
        plt.show()


if __name__ == '__main__':
    unittest.main()
