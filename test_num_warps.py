import matplotlib.pyplot as plt
import torch

from cuda_3d_ops import iterative_3d_warp_cuda, trilinear_splat_cuda


if __name__ == "__main__":
    # make events
    b, d, h, w = 1, 20, 3, 3
    offset, steps = 0, 1
    z = torch.cat([torch.linspace(offset, 1, steps, device="cuda") + i for i in range(d)])
    zi = torch.cat([torch.ones(steps, device="cuda") * i for i in range(d)])
    x, y, val = torch.ones_like(z) * (w // 2), torch.ones_like(z) * (h // 2), torch.ones_like(z)
    events = torch.stack([x, y, z, zi, val], dim=1).view(b, -1, 5)

    # zero flow
    flow = torch.zeros(b, d, h, w, 2, device="cuda")

    # different bases
    contribs = {}
    valids = {}
    bases = [d, d // 2, d // 4]
    for base in bases:
        # warp events
        warped_events = iterative_3d_warp_cuda(events, flow, base)

        # select windows that have everything warped to them
        warped_events = warped_events[:, :, d // 4:-(d // 4)]
        z_ref = torch.arange(d + 1, device="cuda")
        z_ref = z_ref[d // 4:-(d // 4)]

        # unbind
        x, y, z, z_orig, val = warped_events.unbind(dim=-1)
        z_contrib = 1 - (z_ref - z_orig).abs() / min(d, base)
        contribs[base] = z_contrib * val
        valids[base] = val


        # splat
        # warped_events = torch.stack([x, y, z, val * z_contrib], dim=-1).view(b, -1, 4)
        # splatted = trilinear_splat_cuda(warped_events, (d + 1, h, w))

    # plot contributions of single events
    # shape (b, n, d + 1)
    fig, axs = plt.subplots(2, len(contribs), figsize=(12, 6))
    for i, base in enumerate(bases):
        contrib, valid = contribs[base].cpu().numpy(), valids[base].cpu().numpy()
        b, n, _ = contrib.shape
        for j in range(n):
            axs[0, i].plot(contrib[0, j], label=f"event {j}")
            axs[1, i].plot(valid[0, j], label=f"event {j}")
        axs[0, i].set_title(f"contrib base {base}")
        axs[1, i].set_title(f"valid base {base}")
        axs[0, i].grid()
        axs[1, i].grid()
    fig.tight_layout()
    fig.savefig("figures/test_num_warps.png", dpi=300)
