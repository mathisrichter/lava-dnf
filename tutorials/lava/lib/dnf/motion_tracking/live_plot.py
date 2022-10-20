# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

import IPython.display as ipython_display

import bokeh.plotting as bokeh_plotting
import bokeh.io as bokeh_io
import bokeh.models as bokeh_models


class LivePlot(AbstractProcess):
    def __init__(self, shape, up_sampling_factor=1):
        super().__init__(shape=shape, up_sampling_factor=up_sampling_factor)

        self.frame_port = InPort(shape=shape)


@implements(proc=LivePlot, protocol=LoihiProtocol)
@requires(CPU)
class PyTutorialLiveMatrixReaderPMMatplotLib1(PyLoihiProcessModel):
    frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._shape = proc_params["shape"]
        self._up_sampling_factor = proc_params["up_sampling_factor"]

        self._fig = plt.figure(figsize=(5, 5))
        self._ax = self._fig.add_subplot(1, 1, 1)
        print()

    def run_spk(self):
        frame_data = self.frame_port.recv()

        self._ax.clear()
        self._ax.imshow(frame_data)
        self._fig.canvas.draw()


@implements(proc=LivePlot, protocol=LoihiProtocol)
@requires(CPU)
class PyTutorialLiveMatrixReaderPMMatplotLib2(PyLoihiProcessModel):
    frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._shape = proc_params["shape"]
        self._up_sampling_factor = proc_params["up_sampling_factor"]

        self._fig = plt.figure(figsize=(5, 5))
        self._ax = self._fig.add_subplot(1, 1, 1)
        print()

    def run_spk(self):
        frame_data = self.frame_port.recv()

        self._ax.clear()
        self._ax.imshow(frame_data)
        ipython_display.clear_output(wait=True)
        ipython_display.display(self._fig)


@implements(proc=LivePlot, protocol=LoihiProtocol)
@requires(CPU)
class PyTutorialLiveMatrixReaderPMBokeh(PyLoihiProcessModel):
    frame_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._shape = proc_params["shape"]
        self._up_sampling_factor = proc_params["up_sampling_factor"]

        bokeh_io.output_notebook()
        print("")

        a = np.zeros(self._shape)
        p = bokeh_plotting.figure(tooltips=[("x", "$x"), ("y", "$y"),
                                            ("value", "@image")])
        im = p.image(image=[a], x=0, y=0, dw=28, dh=28, palette="Spectral11",
                     level="image")
        self._ds = im.data_source

        color = bokeh_models.LinearColorMapper(palette="Spectral11", low=0, high=1)
        cb = bokeh_models.ColorBar(color_mapper=color, location=(5, 6))
        p.add_layout(cb, 'right')

        ticks = list(range(self._shape[0]))
        p.xaxis[0].ticker = ticks
        p.xgrid[0].ticker = ticks
        p.xgrid.grid_line_color = None
        p.yaxis[0].ticker = ticks
        p.ygrid[0].ticker = ticks
        p.ygrid.grid_line_color = None

        self._target_handle = bokeh_io.show(p, notebook_handle=True)

    def run_spk(self):
        frame_data = self.frame_port.recv()

        self._ds.data["image"] = [frame_data]

        bokeh_io.push_notebook(handle=self._target_handle)
