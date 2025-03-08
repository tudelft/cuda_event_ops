from functools import partial
from pathlib import Path
import time

from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np
import torch

import cuda_event_ops as ceo


if __name__ == "__main__":
    # generate events and flows
    torch.manual_seed(0)
    dtype = torch.float32
    num_events = [1, 10, 100, 1000, 10000, 100000]
    b, d, h, w = 1, 10, 128, 128  # batch, depth (num flow fields), height, width
    repeats = 10
    events_, flows_ = [], []
    for n in num_events:
        # events
        # (b, d, n, 5) tensor with (x, y, z, zi, val) in last dim
        events = torch.rand(b, d, n, 5, device="cuda", dtype=dtype) * torch.tensor(
            [w - 1, h - 1, 1, 1, 1], device="cuda", dtype=dtype
        )
        for i in range(d):
            events[:, i, :, 2] += i  # z
            events[:, i, :, 3] = i  # zi = floor(z)
        events_.append(events)
        # flows
        # (b, d, h, w, 2) tensor with (u, v) flow from z to z+1 in last dim
        flows = torch.rand(b, d, h, w, 2, device="cuda", dtype=dtype)
        flows.requires_grad = True
        flows_.append(flows)

    def once(events, flows, warp_fn, splat_fn):
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        # warp events
        warped_events = warp_fn(events.view(b, -1, 5), flows)
        if warped_events.dim() == 4:  # NOTE: temporary for torch_batch
            warped_events = warped_events[..., [0, 1, 2, 4]].view(b, -1, 4)  # remove z_orig
        torch.cuda.synchronize()
        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        # splat to images
        splatted = splat_fn(warped_events, (d + 1, h, w))
        torch.cuda.synchronize()
        m2 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t2 = time.time()
        # backward loss
        loss = splatted.diff(dim=1).abs().sum()
        loss.backward()
        torch.cuda.synchronize()
        m3 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t3 = time.time()
        # store results
        result_time = {
            "warp": (t1 - t0) * 1000,
            "splat": (t2 - t1) * 1000,
            "backward": (t3 - t2) * 1000,
            "total": (t3 - t0) * 1000,
        }
        result_memory = {
            "warp": m1 - m0,
            "splat": m2 - m0,
            "backward": m3 - m0,
            "total": max(m1 - m0, m2 - m0, m3 - m0),
        }
        loss_val = loss.detach().clone()
        flow_grad = flows.grad.clone()
        # set grad to none
        flows.grad = None
        return result_time, result_memory, loss_val, flow_grad

    # methods
    methods = {
        "torchnaive": (ceo.tn.iterative_3d_warp, ceo.tn.trilinear_splat),
        "torchbatch": (partial(ceo.tb.iterative_3d_warp, num_warps=d), ceo.tb.trilinear_splat),
        "cuda": (partial(ceo.cu.iterative_3d_warp, num_warps=d), ceo.cu.trilinear_splat),
    }

    # benchmark
    results_time, results_memory, losses, grads = DotMap(), DotMap(), DotMap(), DotMap()
    for name, fns in methods.items():
        for n, events, flows in zip(num_events, events_, flows_):
            # max 100 events if naive torch
            if name == "torchnaive" and n > 100:
                continue
            print(f"Running {name} with {n} events")

            # warmup
            warp_fn, splat_fn = fns
            once(events, flows, warp_fn, splat_fn)

            # benchmark
            for i in range(repeats):
                result_time, result_memory, loss, grad = once(events, flows, warp_fn, splat_fn)
                for k, v in result_time.items():
                    results_time[name][k][n] += [v]
                for k, v in result_memory.items():
                    results_memory[name][k][n] += [v]
                losses[n] += [loss]
                grads[n] += [grad]

    # check loss and grad equality
    for n in num_events:
        losses_eq, grads_eq = [], []
        losses_diff, grads_diff = [], []
        for l0, l1, g0, g1 in zip(losses[n][:-1], losses[n][1:], grads[n][:-1], grads[n][1:]):
            losses_eq.append(torch.allclose(l0, l1))
            grads_eq.append(torch.allclose(g0, g1))
            losses_diff.append(torch.max(torch.abs(l0 - l1)))
            grads_diff.append(torch.max(torch.abs(g0 - g1)))

        print(f"Losses all equal for {n} events: {all(losses_eq)}, largest diff: {max(losses_diff)}")
        print(f"Grads all equal for {n} events: {all(grads_eq)}, largest diff: {max(grads_diff)}")

    # print results
    for name, result in results_time.items():
        print(name)
        for k, v in result.items():
            print(k)
            for n, ts in v.items():
                print(f"{n} events: {np.median(ts):.3f} ms")
        print()
    for name, result in results_memory.items():
        print(name)
        for k, v in result.items():
            print(k)
            for n, mem in v.items():
                print(f"{n} events: {np.median(mem) / (1024**2):.3f} MB")
        print()

    # write results to separate files
    folder = Path(__file__).parent / "benchmark_warp_events_results_4090"
    folder.mkdir(exist_ok=True, parents=True)
    for name, result in results_time.items():
        for k, v in result.items():
            with open(folder / f"{name}_{k}_runtime.csv", "w") as f:
                f.write("num_events,time_ms\n")
                for n, ts in v.items():
                    f.write(f"{n},{np.median(ts):.3f}\n")
    for name, result in results_memory.items():
        for k, v in result.items():
            with open(folder / f"{name}_{k}_memory.csv", "w") as f:
                f.write("num_events,memory_mb\n")
                for n, mem in v.items():
                    f.write(f"{n},{np.median(mem) / (1024**2):.3f}\n")

    # plot results
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharey="row", sharex="col")
    for name, result in results_time.items():
        for i, (k, v) in enumerate(result.items()):
            x_line = np.array(list(v.keys()))
            y_line = np.array([np.median(times) for times in v.values()])
            y_25 = np.array([np.percentile(times, 25) for times in v.values()])
            y_75 = np.array([np.percentile(times, 75) for times in v.values()])
            axs[0, i].fill_between(x_line, y_25, y_75, alpha=0.3)
            axs[0, i].plot(x_line, y_line, label=name.replace("_", " "))
            axs[0, i].set_title(k)
            axs[0, i].set_xscale("log")
            axs[0, i].set_yscale("log")
            axs[0, i].grid(True)
    for name, result in results_memory.items():
        for i, (k, v) in enumerate(result.items()):
            x_line = np.array(list(v.keys()))
            y_line = np.array([np.median(times) / (1024**2) for times in v.values()])
            y_25 = np.array([np.percentile(times, 25) / (1024**2) for times in v.values()])
            y_75 = np.array([np.percentile(times, 75) / (1024**2) for times in v.values()])
            axs[1, i].fill_between(x_line, y_25, y_75, alpha=0.3)
            axs[1, i].plot(x_line, y_line, label=name.replace("_", " "))
            axs[1, i].set_xlabel("num events/bin")
            axs[1, i].set_xscale("log")
            axs[1, i].set_yscale("log")
            axs[1, i].grid(True)
    # axs[0, -1].axvline(2119, color="black", linestyle="--")  # uzhfpv
    # axs[0, -1].axvline(6230, color="black", linestyle="--")  # cz
    # axs[0, -1].axvline(5867, color="black", linestyle="--")  # mvsec
    # axs[0, -1].axvline(150797, color="black", linestyle="--")  # dsec
    axs[0, 0].legend()
    axs[0, 0].set_ylabel("runtime [ms]")
    axs[1, 0].set_ylabel("peak delta memory [MB]")
    fig.suptitle("Warping and splatting events: torch vs cuda", fontweight="bold", fontsize=18)
    fig.tight_layout()
    plt.savefig(Path(__file__).parent / "benchmark_warp_events_4090.pdf", bbox_inches="tight", transparent=True)
