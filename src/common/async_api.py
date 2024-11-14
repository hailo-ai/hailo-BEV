# pylint: disable=E0401
import queue
from functools import partial
from loguru import logger
import numpy as np
from hailo_platform import (HEF, VDevice, FormatType, HailoSchedulingAlgorithm)


def create_vdevice_params():
    """
    Create parameters for a virtual device.
    """
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.multi_process_service = False
    return params

class HailoAsyncInference:
    def __init__(self, target, hef_path, queue_out, demo,
                 input_names, output_names, batch_size=1, output_type='FLOAT32', input_type='FLOAT32'):
        """
        Initialize the HailoAsyncInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
            batch_size (int): Batch size for inference.
            output_type (str): Format type of the output stream.
        """

        self.hef = HEF(hef_path)
        self.demo = demo
        self.target = target
        self.input_names = input_names
        self.output_names = output_names
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)
        self.input_type = input_type
        self._set_input_output(output_type, input_type)
        self.input_vstream_info, self.output_vstream_info = self._get_vstream_info()
        self.configured_infer_model = self.infer_model.configure()
        self.queue = queue_out
        self.outputs = []
        self.batch_size = batch_size


    def _set_input_output(self, output_type, input_type):
        """
        Set the input and output layer information for the HEF model.

        Args:
            output_type (str): Format type of the output stream.
        """
        for name in self.input_names:
            self.infer_model.input(name).set_format_type(getattr(FormatType, input_type))
        for name in self.output_names:
            self.infer_model.output(name).set_format_type(getattr(FormatType, output_type))

    def callback(self, completion_info, all_bindings, out_queue):
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the inference task.
            bindings: Bindings object containing input and output buffers.
        """

        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            self.outputs = {}
            for name in self.output_names:
                self.outputs[name] = []
                for binding in all_bindings:
                    self.outputs[name].append(binding.output(name).get_buffer())

            while not self.demo.get_terminate():
                try:
                    out_queue.put(self.outputs, block=True, timeout=0.5)
                    break
                except queue.Full:
                    pass


    def _get_vstream_info(self):
        """
        Get information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        return self.hef.get_input_vstream_infos(), self.hef.get_output_vstream_infos()

    def run(self, input_data):
        """
        Run asynchronous inference on the Hailo-8 device.

        Args:
            input_data (np.ndarray): Input data for inference.

        Returns:
            list: List of inference outputs.
        """

        all_bindings = []
        for batch in range(self.batch_size):
            bindings = self._create_bindings()
            for name in self.input_names:
                bindings.input(name).set_buffer(input_data[name][batch])

            all_bindings.append(bindings)

        self.configured_infer_model.wait_for_async_ready(timeout_ms=1000)

        job = self.configured_infer_model.run_async(all_bindings,
                            partial(self.callback, all_bindings=all_bindings, out_queue=self.queue))

        return job

    def _create_bindings(self):
        """
        Create bindings for input and output buffers.

        Returns:
            bindings: Bindings object with input and output buffers.
        """
        output_buffers = {name: np.empty(self.infer_model.output(name).shape, dtype=np.float32)
                          for name in self.infer_model.output_names}
        return self.configured_infer_model.create_bindings(output_buffers=output_buffers)
