import torch
from src.models.patchTST import PatchTST

def main():
    # Example configuration dictionary
    configs = {
        'enc_in': 1, # no of input features/channels
        'target_dim': 1, # no of output features/channels
        'patch_len': 512, # length of each patch
        'stride': 256, # stride for patching
        'num_patch': 4, # number of patches
        'n_layers': 6, # number of transformer layers
        'd_model': 512, # dimension of the model
        'n_heads': 8, # number of attention heads
        'shared_embedding': True, # whether to use shared embedding
        'd_ff': 2048, # dimension of the feed-forward network
        'dropout': 0.1, # dropout rate
        'head_dropout': 0.1, # dropout rate for attention heads
        'act': 'relu',
        'head_type': 'pretrain', # Possible values: 'pretrain', 'prediction', 'classification', 'regression'
        'res_attention': False, # whether to use residual attention
    }
    # access configs as an object
    class Configs:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    configs = Configs(configs)
    
    # Initialize the PatchTST model
    model = PatchTST(
        c_in=configs.enc_in,
        target_dim=configs.target_dim,
        patch_len=configs.patch_len,
        stride=configs.stride,
        num_patch=configs.num_patch,
        n_layers=configs.n_layers,
        n_heads=configs.n_heads,
        d_model=configs.d_model,
        shared_embedding=True,
        d_ff=configs.d_ff,
        dropout=configs.dropout,
        head_dropout=configs.head_dropout,
        act=configs.act,
        head_type=configs.head_type,
        res_attention=configs.res_attention
    )
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("PatchTST model initialized successfully.")
    output = model.forward(torch.randn(1, configs.num_patch, configs.enc_in, configs.patch_len))
    print("Model output shape:", output.shape)


if __name__ == "__main__":
    main()