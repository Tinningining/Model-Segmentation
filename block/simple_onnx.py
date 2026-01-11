import torch
import torch.nn as nn
import onnx


# 1. 定义一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def main():
    # 2. 实例化模型
    model = SimpleModel()
    model.eval()

    # 3. 创建固定 shape 的输入
    # 静态输入: batch=1, feature=4
    dummy_input = torch.randn(1, 4)

    # 4. 导出 ONNX（静态输入）
    onnx_path = "static_input_example.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
        # ⚠️ 不要写 dynamic_axes
    )

    print(f"ONNX 已导出到: {onnx_path}")

    # 5. 校验 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型校验通过")

    # 6. 打印输入 / 输出 shape，确认是静态的
    print("\nONNX 输入信息:")
    for inp in onnx_model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        shape = [d.dim_value for d in dims]
        print(f"  {inp.name}: {shape}")

    print("\nONNX 输出信息:")
    for out in onnx_model.graph.output:
        dims = out.type.tensor_type.shape.dim
        shape = [d.dim_value for d in dims]
        print(f"  {out.name}: {shape}")


if __name__ == "__main__":
    main()
