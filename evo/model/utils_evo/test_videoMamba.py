from VideoMamba import videomamba_tiny
import torch

def main():
    vm = videomamba_tiny(
            embed_dim=192, # final number of channels.
            channels=16, # which is the number of generated seqences by evo... 
            # num_classes=1000, # yes, now I am going to use the cls token to determine which one is the best next token....right? or should I go by a distribution after the softmax...
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            bimamba=True,
            # video
            kernel_size=1, 
            num_tokens=64, # length of each tokenized version of the 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0)
    # testing the input
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vm = vm.to(device)
    test_input = torch.randn(16, 16, 64).to(device)
    output = vm(test_input)
    print(output.shape)


if __name__ == '__main__':
    main()