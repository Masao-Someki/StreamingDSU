import torch
from transformers import AutoModel
from calflops.calculate_pipline import CalFlopsPipline
from calflops.utils import get_module_flops


if __name__ == "__main__":
    device = 0
    model = AutoModel.from_pretrained("microsoft/wavlm-large").to(device)

    for layer in range(25):
        full_window = 3000  # 30s input
        x = torch.ones(()).new_empty((1, 80 + 320 * full_window, ), dtype=next(model.parameters()).dtype, device=device)
        pipeline = CalFlopsPipline(model=model, include_backPropagation=False, compute_bp_factor=2.0)

        pipeline.start_flops_calculate(ignore_list=[])
        model(x)
        total_flops = pipeline.get_total_flops()
        attn_flops = sum(
            get_module_flops(layer.attention)
            for layer in model.encoder.layers
        )
        pipeline.end_flops_calculate()

        for window in (full_window, 1, 2, 3, 5, 9, 17, 33, 65, 129, ):
            print(total_flops - attn_flops + attn_flops / full_window * window, end="\t")

        if (24 - layer) == 21:
            print("\nFull past:", end="\t")
            for window in (1, 2, 3, 5, 9, 17, 33, 65, 129, ):
                diag = attn_flops / full_window * window
                lower_tri = (attn_flops - diag) / 2
                print(total_flops - attn_flops + diag + lower_tri, end="\t")

        if len(model.encoder.layers):
            del model.encoder.layers[-1]
        print()
