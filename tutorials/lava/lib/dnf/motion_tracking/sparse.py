# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.nc.type import LavaNcType

try:
    from lava.magma.core.model.nc.net import NetL2
except ImportError:
    class NetL2:
        pass
from lava.magma.compiler.node import Node
from lava.magma.core.resources import NeuroCore
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.nc.model import AbstractNcProcessModel
from lava.magma.core.model.nc.var import NcVar
from lava.proc.dense.process import Dense
from lava.utils.weightutils import SignMode, optimize_weight_bits, \
    truncate_weights, determine_sign_mode, clip_weights
from lava.magma.core.model.nc.tables import Nodes
from scipy.sparse import csr_matrix, find


import numpy as np
import typing as ty

from lava.magma.core.process.connection import ConnectionProcess
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class Sparse(AbstractProcess):
    def __init__(self,
                 *,
                 weights: np.ndarray,
                 name: ty.Optional[str] = None,
                 num_message_bits: ty.Optional[int] = 0,
                 log_config: ty.Optional[LogConfig] = None,
                 **kwargs) -> None:

        super().__init__(weights=weights,
                         num_message_bits=num_message_bits,
                         name=name,
                         log_config=log_config,
                         **kwargs)

        self._validate_weights(weights)
        shape = weights.shape

        # Ports
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))

        # Variables
        self.weights = Var(shape=shape, init=weights)
        self.a_buff = Var(shape=(shape[0],), init=0)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)

    @staticmethod
    def _validate_weights(weights: np.ndarray) -> None:
        if len(np.shape(weights)) != 2:
            raise ValueError("Dense Process 'weights' expects a 2D matrix, "
                             f"got {weights}.")


@implements(proc=Sparse, protocol=LoihiProtocol)
@requires(NeuroCore)
class NcModelDense(AbstractNcProcessModel):
    """Implements behavior of a DENSE connection process for Loihi 2"""
    # Declare port implementation
    s_in: NcInPort = LavaNcType(NcInPort, np.int32, precision=24)
    a_out: NcOutPort = LavaNcType(NcOutPort, np.int16, precision=16)
    # Declare variable implementation
    a_buff: NcVar = LavaNcType(NcVar, np.int32, precision=16)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    weights: NcVar = LavaNcType(NcVar, np.int32, precision=8)
    num_message_bits: NcVar = LavaNcType(NcVar, np.int8, precision=8)

    def allocate(self, net: NetL2):
        """Allocate resources for Dense connections."""
        # Obtain the shape of input and output
        input_shape = self.s_in.shape
        output_shape = self.a_out.shape
        process_size = (np.prod(output_shape), np.prod(input_shape))
        weight_exp: int = self.proc_params.get("weight_exp", 0)
        num_weight_bits: int = self.proc_params.get("num_weight_bits", 8)
        sign_mode: SignMode = self.proc_params.get("sign_mode", None)
        weights: np.ndarray = self.weights.var.get()
        sign_mode = sign_mode or determine_sign_mode(weights)
        weights = clip_weights(weights, sign_mode, num_bits=8)
        weights = truncate_weights(weights, sign_mode, num_weight_bits)
        optimized_weights = optimize_weight_bits(
            weights=weights,
            sign_mode=sign_mode,
            loihi2=True
        )
        weights: np.ndarray = optimized_weights.weights
        weights_exp: int = optimized_weights.weight_exp + weight_exp
        num_weight_bits = optimized_weights.num_weight_bits

        sparse_weights = csr_matrix(weights)
        dst, src, wgt = find(sparse_weights)

        # Sort sparse weight in the order of input dimension
        idx = np.argsort(src)
        src = src[idx]
        dst = dst[idx]
        wgt = wgt[idx]
        # Allocate Input Axons
        ax_in: Nodes = net.ax_in.allocate(shape=input_shape)
        ax_in_cfg: Nodes = net.ax_in_cfg.allocate(
            shape=1,
            num_message_bits=self.num_message_bits.var.get())
        # Allocate Synapses
        syn: Nodes = net.syn.allocate(shape=wgt.shape,
                                      weights=wgt,
                                      delays=0)
        syn_cfg: Nodes = net.syn_cfg.allocate(
            shape=1,
            is_signed=int(sign_mode == SignMode.MIXED),
            num_weight_bits=num_weight_bits,
            num_delay_bits=0,
            weights_exp=weights_exp)
        # Allocate dendritic accumulators
        dend_acc_cfg: Nodes = net.dend_acc_cfg.allocate(shape=1,
                                                        num_delay_bits=0)
        dend_acc: Nodes = net.dend_acc.allocate(shape=(process_size[0],))
        # Connect InPort of Process to neurons
        self.s_in.connect(ax_in)
        # Connect Nodes
        ax_in.connect(ax_in_cfg)
        ax_in[src].connect(syn)
        syn.connect(syn_cfg)
        syn.connect(dend_acc[dst])
        # Connect output axon to OutPort of Process
        dend_acc.connect(dend_acc_cfg)
        dend_acc.connect(self.a_out)