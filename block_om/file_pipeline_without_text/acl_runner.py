import acl
import sys
import numpy as np


class ACLModel:
    def __init__(self, model_path, device_id=0):
        self.model_path = model_path
        self.device_id = device_id

    def init(self):
        self._check(acl.init(), "acl.init")
        self._check(acl.rt.set_device(self.device_id), "set_device")
        self.context, ret = acl.rt.create_context(self.device_id)
        self._check(ret, "create_context")
        self.stream, ret = acl.rt.create_stream()
        self._check(ret, "create_stream")

        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        self._check(ret, "load model")

        self.desc = acl.mdl.create_desc()
        self._check(acl.mdl.get_desc(self.desc, self.model_id), "get_desc")

    def execute(self, inputs: list[np.ndarray]):
        input_ds = acl.mdl.create_dataset()
        output_ds = acl.mdl.create_dataset()

        # inputs
        for arr in inputs:
            size = arr.nbytes
            dev, ret = acl.rt.malloc(size, 0)
            self._check(ret, "malloc input")
            self._check(
                acl.rt.memcpy(dev, size, arr.ctypes.data, size, 1),
                "memcpy H2D",
            )
            buf = acl.create_data_buffer(dev, size)
            acl.mdl.add_dataset_buffer(input_ds, buf)

        # outputs
        for i in range(acl.mdl.get_num_outputs(self.desc)):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            dev, ret = acl.rt.malloc(size, 0)
            self._check(ret, "malloc output")
            buf = acl.create_data_buffer(dev, size)
            acl.mdl.add_dataset_buffer(output_ds, buf)

        self._check(
            acl.mdl.execute(self.model_id, input_ds, output_ds),
            "execute",
        )

        outputs = []
        for i in range(acl.mdl.get_num_outputs(self.desc)):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            host = np.empty(size, dtype=np.uint8)
            dev = acl.get_data_buffer_addr(
                acl.mdl.get_dataset_buffer(output_ds, i)
            )
            self._check(
                acl.rt.memcpy(host.ctypes.data, size, dev, size, 2),
                "memcpy D2H",
            )
            outputs.append(host)

        acl.mdl.destroy_dataset(input_ds)
        acl.mdl.destroy_dataset(output_ds)
        return outputs

    def finalize(self):
        acl.mdl.unload(self.model_id)
        acl.mdl.destroy_desc(self.desc)
        acl.rt.destroy_stream(self.stream)
        acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

    def _check(self, ret, msg):
        if ret != 0:
            print(f"[ERROR] {msg}, ret={ret}")
            err = acl.get_recent_err_msg()
            if err:
                print("ACL:", err)
            sys.exit(1)
