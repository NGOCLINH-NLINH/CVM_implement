import torch
import torch.nn.functional as F
import numpy as np


def get_conv_modules(model):
    conv_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_modules.append((name, module))
    return conv_modules


def register_hooks(conv_modules):
    hooks = []

    def hook(module, input, output):
        module.act = input[0].detach()

    for name, module in conv_modules:
        hooks.append(module.register_forward_hook(hook))
    return hooks


def get_representation_matrix(model, conv_modules, device, train_loader, num_samples=128):
    model.eval()
    example_data = []
    samples_collected = 0

    for images, _, _ in train_loader:
        example_data.append(images)
        samples_collected += images.size(0)
        if samples_collected >= num_samples:
            break
    example_data = torch.cat(example_data, dim=0)[:num_samples].to(device)

    with torch.no_grad():
        _ = model(example_data)

    mat_list = []
    for name, module in conv_modules:
        act = module.act
        k = module.kernel_size[0]
        s = module.stride[0]
        p = module.padding[0]
        d = module.dilation[0]

        unfolded = F.unfold(act, kernel_size=k, stride=s, padding=p, dilation=d)
        b, dim, L = unfolded.shape

        mat = unfolded.transpose(1, 2).contiguous().view(-1, dim).cpu().numpy()
        mat_list.append(mat.T)

        module.act = None

    return mat_list


def update_GPM(mat_list, threshold, feature_list=[]):
    if not feature_list:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold)
            feature_list.append(U[:, 0:r])
            print(f"  -> Layer {i}: Lock {r}/{U.shape[0]} dims")
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U_old = feature_list[i]

            act_hat = activation - np.dot(np.dot(U_old, U_old.T), activation)

            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

            _, S_ori, _ = np.linalg.svd(activation, full_matrices=False)
            sval_total_ori = (S_ori ** 2).sum()
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total_ori

            accumulated_sval = (sval_total_ori - sval_hat) / sval_total_ori
            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break

            if r > 0:
                Ui = np.hstack((feature_list[i], U[:, 0:r]))
                feature_list[i] = Ui[:, 0:Ui.shape[0]] if Ui.shape[1] > Ui.shape[0] else Ui
            print(f"  -> Layer {i}: Lock {feature_list[i].shape[1]}/{feature_list[i].shape[0]} dims")

    return feature_list