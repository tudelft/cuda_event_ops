import time

from dotmap import DotMap
import numpy as np
import torch
import torch.nn.functional as F

import cuda_3d_ops as c3o


def pad_sequence(sequences, batch_first=False):
    """
    Args:
        sequences: N-size list of (b, c, ?) tensors with variable length (?).
        batch_first: bool indicating whether to return (b, N, 4, ?) or (N, b, 4, ?) tensor.
    Returns:
        (b, N, 4, ?) or (N, b, 4, ?) tensor of padded sequences.
    """

    transposed = [seq.transpose(0, 2) for seq in sequences]  # variable length first
    padded = torch.nn.utils.rnn.pad_sequence(transposed, batch_first=batch_first)
    return padded.transpose(0, 3) if not batch_first else padded.transpose(1, 3)


def event_warping(idx, flow, ts, t_ref):
    return idx + (t_ref - ts) * flow


def linear_warp(events, flow, t_ref, keep_ts=True):
    """
    Warp events linearly to reference time.
    Changes events in-place, i.e. does not copy data.

    Args:
        events: (*, b, 4, n) tensor of events with (ts, y, x, p) in -2 dim.
        flow: (*, b, 2, n) tensor of (y, x) flow for each event.
        t_ref: (*, 1, 1, 1) tensor or scalar of reference time to warp to.

    Returns:
        (*, b, 4, n) tensor of warped events.
    """
    events[..., 1:3, :] += (t_ref - events[..., 0:1, :]) * flow
    if not keep_ts:
        events[..., 0:1, :] = t_ref
    return events


def get_event_flow_3d(events, flow):
    """
    Args:
        events: (d, b, 4, n) tensor of events with (ts, y, x, p) in -2 dim.
        flow: (d, b, 2, h, w) tensor of (y, x) flow maps.

    Returns:
        (d, b, 2, n) tensor of (y, x) flow for each event.
    """

    # discretize ts (no bilinear in time)
    # grid: order (x, y, t) in last dim and normalize to (-1, 1)
    d, _, _, h, w = flow.shape
    y, x = events[..., 1:3, :].unbind(-2)
    z = torch.ones_like(x) * torch.arange(d, dtype=events.dtype, device=events.device).view(-1, 1, 1)
    resolution = torch.tensor([w, h, d], dtype=events.dtype, device=events.device)  # needs clamp if d == 1
    grid_norm = 2 * torch.stack([x, y, z], dim=-1).unsqueeze(3) / (resolution - 1).clamp(min=1) - 1

    # use grid_sample so idx can be float
    # align corners because warping to pixel centers
    # flow shape: (b, 2, d, n, w); idx_norm shape: (b, d, n, 1, 3)
    # returns (b, 2, d, n, 1)
    flow = F.grid_sample(flow.permute(1, 2, 0, 3, 4), grid_norm.transpose(0, 1), mode="bilinear", align_corners=True)

    return flow.squeeze(-1).permute(2, 0, 1, 3)


def compute_inside_mask(idx, resolution):
    """
    Args:
        idx: (*, b, 2, n) tensor of (y, x) event indices in -2 dim.
        resolution: (2,) tensor of (h, w) resolution.

    Returns:
        (*, b, 1, n) mask of events inside resolution.
    """
    # in taming: idx >= 0 & idx <= resolution - 1
    # ensures enough space for a whole pixel
    # < resolution is fine if pixel is int but doesn't work if pixel can be float
    mask = (idx >= 0) & (idx <= resolution.unsqueeze(1) - 1)
    return mask.all(dim=-2, keepdim=True)


def iterative_3d_warp_torch_batch(events, flows, base):
    """
    Args:
        events: (d, b, 4, n) tensor of events with (ts, y, x, p) in -2 dim.
        flows: (d, b, 2, h, w) tensor of (y, x) flow maps.
        base: int of number of neighboring bins (one-sided) to consider for each reference time.

    Returns:
        torch.Tensor: A tensor of shape (d + 1, b, 4, n) with the warped events.
    """

    # get deblurring window and resolution
    d, _, _, h, w = flows.shape
    resolution = torch.tensor([h, w], device=flows.device)

    # repeat for simultaneous fw and bw warping
    events = events.repeat(2, 1, 1, 1)  # copies

    # lists to store warped events, ts and inside mask
    # we need to know where they came from + where they're going
    warped_events = [[None for _ in range(d)] for _ in range(d + 1)]
    warped_mask_inside = [None for _ in range(d)]

    # warping stages
    for i in range(d):
        # bin indices to take events/flow from (fw, bw warp)
        # we keep bin position when warping (only change index of events)
        t0 = torch.arange(d - i, device=events.device)
        t1 = torch.arange(i, d, device=events.device)
        t_events = torch.cat([t0, t1 + d])
        t_flow = torch.cat([t1, t0])
        t_ref = torch.cat([t1 + 1, t0])

        # select events, flow and original timestamps (for scaling)
        # advanced indexing copies
        select_events = events[t_events]
        select_flow = flows[t_flow]

        # sample flow that will warp events at event locations
        # ensure integer t because we don't want bilinear there
        # (is a concat of fw/bw, not chronological)
        select_event_flow = get_event_flow_3d(select_events, select_flow)

        # fw/bw warp events to reference time
        # overwrite ts with reference time
        # all in-place
        select_events = linear_warp(select_events, select_event_flow, t_ref.view(-1, 1, 1, 1), keep_ts=False)

        # discard events warped outside image
        mask_inside = compute_inside_mask(select_events[..., 1:3, :], resolution)
        select_events[..., 1:4, :] *= mask_inside  # mask idx and p, can be done in-place

        # save warped events, no copy here
        events[t_events] = select_events

        # save warped events, ts and inside mask for border compensation
        # also take into account mask outside base
        for j, (src, dst) in enumerate(zip([*t0, *t1], t_ref)):
            warped_events[dst][src] = select_events[j : j + 1] if i < base else None
            warped_mask_inside[src] = (
                warped_mask_inside[src] * mask_inside[j : j + 1]
                if warped_mask_inside[src] is not None
                else mask_inside[j : j + 1]
            )

    # concat at each reference time
    # do border compensation
    # events that go out anywhere are masked everywhere
    for i in range(d + 1):
        warped_events[i] = torch.cat([e * m for e, m in zip(warped_events[i], warped_mask_inside) if e is not None])

    return torch.stack(warped_events)


def inv_l1_dist_prod(idx):
    """
    Args:
        idx: (*, b, 2, n) tensor of (y, x) event indices in -2 dim.

    Returns:
        (*, b, 2, 4n) tensor of nearest integer indices.
        (*, b, 1, 4n) tensor of inverse L1 distances.
    """

    # 1d rounded sides
    # doing idx.floor() + 1 is different from doing (idx.floor() + 1)
    # due to floating point precision; event_flow does latter so keep that
    # ceil is different from floor + 1 for integers
    top_y = idx[..., 0, :].floor()
    bot_y = (idx[..., 0, :] + 1).floor()
    left_x = idx[..., 1, :].floor()
    right_x = (idx[..., 1, :] + 1).floor()

    # combine to 2d corners
    top_left = torch.stack([top_y, left_x], dim=-2)
    top_right = torch.stack([top_y, right_x], dim=-2)
    bottom_left = torch.stack([bot_y, left_x], dim=-2)
    bottom_right = torch.stack([bot_y, right_x], dim=-2)

    # corners sequentially in event dimension
    idx_round = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=-1)
    idx = idx.repeat(1, 1, 4) if idx.dim() == 3 else idx.repeat(1, 1, 1, 4)

    # compute inverse l1 distances
    # event_flow uses max(0, 1 - abs(idx - idx_round)), but that kills gradients at bounds
    # clamp doesn't: https://github.com/pytorch/pytorch/issues/7002, should be better
    inv_dist = (1 - (idx - idx_round).abs()).clamp(min=0)

    # weight = product of distances
    weight = inv_dist.prod(dim=-2, keepdim=True)

    return idx_round, weight


def accumulate_to_image(idx, weight, mask, resolution):
    """
    Args:
        idx: (*, b, 2, n) tensor of (y, x) event indices in -2 dim.
        weight: (*, b, 1, n) tensor of weights to accumulate.
        mask: (*, b, 1, n) mask tensor.
        resolution: (2,) tensor of (h, w) resolution.

    Returns:
        (*, b, 1, h, w) tensor of accumulated images.
    """

    # 1d image to accumulate to
    image = torch.zeros(*idx.shape[:-2], 1, resolution.prod(), dtype=idx.dtype, device=idx.device)

    # convert idx to 1d
    idx = (idx[..., 0:1, :] * resolution[1] + idx[..., 1:2, :]).long()

    # accumulate and reshape to 2d
    image.scatter_add_(-1, idx, weight * mask)
    # image = scatter_add(weight * mask, idx, dim=-1)
    return image.unflatten(-1, resolution.tolist())


def bilinear_splat_torch_batch(events, grid_resolution):
    # get deblurring window and resolution
    _, _, b, _, _ = events.shape
    d, h, w = grid_resolution
    resolution = torch.tensor([h, w], device=events.device)

    # iwe and iwt at each reference time
    iwe = torch.zeros(d + 1, b, 1, h, w, dtype=events.dtype, device=events.device)

    # build iwe and iwt
    for i in range(d + 1):
        # inverse product of l1 distances to closest pixels
        corners, weights = inv_l1_dist_prod(events[i][..., 1:3, :])

        # discard corners outside image
        mask_inside = compute_inside_mask(corners, resolution)
        corners = corners * mask_inside
        weights = weights * mask_inside

        # iwe
        values = events[i][..., 3:4, :].repeat(1, 1, 1, 4)  # repeat for corners
        iwe[i] += accumulate_to_image(corners, weights * values, torch.ones_like(weights), resolution).sum(0)

    return iwe


def iterative_3d_warp_bilinear_splat_torch_combined(events, flow, base):
    """
    Args:
        events: (d, b, 4, n) tensor of events with (ts, y, x, p) in -2 dim.
        flow: (d, b, 2, h, w) tensor of (y, x) flow maps.
        base: int of number of neighboring bins (one-sided) to consider for each reference time.

    Returns:
        (d + 1, b, 1, h, w) tensor of images of warped events.
    """

    # get deblurring window and resolution
    d, _, _, h, w = flow.shape
    resolution = torch.tensor([h, w], device=flows.device)

    # repeat for simultaneous fw and bw warping
    events = events.repeat(2, 1, 1, 1)  # copies

    # iwe and iwt at each reference time
    iwe = torch.zeros(d + 1, b, 1, h, w, dtype=events.dtype, device=events.device)

    # warping stages
    for i in range(d):
        # bin indices to take events/flow from (fw, bw warp)
        # we keep bin position when warping (only change index of events)
        t0 = torch.arange(d - i, device=events.device)
        t1 = torch.arange(i, d, device=events.device)
        t_events = torch.cat([t0, t1 + d])
        t_flow = torch.cat([t1, t0])
        t_ref = torch.cat([t1 + 1, t0])

        # if in base: warp events
        # if base < d we don't get good iwes at 0 and end for this
        if i < base:
            # select events, flow and original timestamps (for scaling)
            # advanced indexing copies
            select_events = events[t_events]
            select_flow = flow[t_flow]

            # sample flow that will warp events at event locations
            # ensure integer t because we don't want bilinear there
            # (is a concat of fw/bw, not chronological)
            select_event_flow = get_event_flow_3d(select_events, select_flow)

            # fw/bw warp events to reference time
            # overwrite ts with reference time
            # all in-place
            select_events = linear_warp(select_events, select_event_flow, t_ref.view(-1, 1, 1, 1), keep_ts=False)

            # discard events warped outside image
            mask_inside = compute_inside_mask(select_events[..., 1:3, :], resolution)
            select_events[..., 1:4, :] *= mask_inside  # mask idx and p, can be done in-place

            # save warped events, no copy here
            events[t_events] = select_events

            # put warped events in iwe

            # inverse product of l1 distances to closest pixels
            corners, weights = inv_l1_dist_prod(select_events[..., 1:3, :])

            # discard corners outside image
            mask_inside = compute_inside_mask(corners, resolution)
            corners = corners * mask_inside
            weights = weights * mask_inside

            # accumulate to images
            # done in two steps because else no summing due to duplicate indices
            # neg and pos cannot be merged because of 0 polarity due to padding
            t_ref_fw = i + 1
            t_ref_bw = d - i

            # split into fw and bw views
            corners_fw, corners_bw = corners.chunk(2)
            weights_fw, weights_bw = weights.chunk(2)

            # iwe
            values = select_events[..., 3:4, :].repeat(1, 1, 1, 4)  # repeat for corners
            values_fw, values_bw = values.chunk(2)
            iwe[t_ref_fw:] += accumulate_to_image(
                corners_fw, weights_fw * values_fw, torch.ones_like(weights_fw), resolution
            )
            iwe[:t_ref_bw] += accumulate_to_image(
                corners_bw, weights_bw * values_bw, torch.ones_like(weights_bw), resolution
            )

    return iwe


"""
TODO:
- Why quite large grad differences for large number of events? Could be check for inside? <= vs <?
- Add combined method for cuda? Small differences for combined torch but maybe worth it
"""
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

    # naive torch
    def torch_naive_once(events, flows):
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        # warp events
        warped_events = c3o.tn.iterative_3d_warp(events.view(b, -1, 5), flows)
        warped_events = warped_events[..., [0, 1, 2, 4]].view(b, -1, 4)  # remove z_orig
        torch.cuda.synchronize()
        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        # splat to images
        splatted = c3o.tn.trilinear_splat(warped_events, (d + 1, h, w))
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
        result_time = dict(
            warp=(t1 - t0) * 1000, splat=(t2 - t1) * 1000, backward=(t3 - t2) * 1000, total=(t3 - t0) * 1000
        )
        result_memory = dict(warp=m1 - m0, splat=m2 - m0, backward=m3 - m0, total=max(m1 - m0, m2 - m0, m3 - m0))
        loss_val = loss.detach().clone()
        flow_grad = flows.grad.clone()
        # set grad to none
        flows.grad = None
        return result_time, result_memory, loss_val, flow_grad

    # # batched torch
    # def torch_batch_once(events, flows):
    #     events_p = events[..., [2, 1, 0, 4]].permute(1, 0, 3, 2).contiguous()
    #     flows_p = flows[..., [1, 0]].permute(1, 0, 4, 2, 3).contiguous()
    #     torch.cuda.synchronize()
    #     m0 = torch.cuda.memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     t0 = time.time()
    #     # warp events
    #     warped_events = iterative_3d_warp_torch_batch(events_p, flows_p, d)
    #     torch.cuda.synchronize()
    #     m1 = torch.cuda.max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     t1 = time.time()
    #     # splat to images
    #     splatted = bilinear_splat_torch_batch(warped_events, (d, h, w))  # d+1 inside
    #     splatted = splatted.squeeze(2).permute(1, 0, 2, 3).contiguous()
    #     torch.cuda.synchronize()
    #     m2 = torch.cuda.max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     t2 = time.time()
    #     # backward loss
    #     loss = splatted.diff(dim=1).abs().sum()
    #     loss.backward()
    #     torch.cuda.synchronize()
    #     m3 = torch.cuda.max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     t3 = time.time()
    #     # store results
    #     result_time = DotMap(
    #         warp=(t1 - t0) * 1000, splat=(t2 - t1) * 1000, backward=(t3 - t2) * 1000, total=(t3 - t0) * 1000
    #     )
    #     result_memory = DotMap(warp=m1 - m0, splat=m2 - m0, backward=m3 - m0, total=max(m1 - m0, m2 - m0, m3 - m0))
    #     loss_val = loss.detach().clone()
    #     flow_grad = flows.grad.clone()
    #     # set grad to none
    #     flows.grad = None
    #     return result_time, result_memory, loss_val, flow_grad

    # # combined torch
    # # no border compensation
    # def torch_combined_once(events, flows):
    #     events_p = events[..., [2, 1, 0, 4]].permute(1, 0, 3, 2).contiguous()
    #     flows_p = flows[..., [1, 0]].permute(1, 0, 4, 2, 3).contiguous()
    #     torch.cuda.synchronize()
    #     m0 = torch.cuda.memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     t0 = time.time()
    #     # combined warp and splat events to images
    #     splatted = iterative_3d_warp_bilinear_splat_torch_combined(events_p, flows_p, d)
    #     torch.cuda.synchronize()
    #     m1 = torch.cuda.max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     # backward loss
    #     loss = splatted.diff(dim=1).abs().sum()
    #     loss.backward()
    #     torch.cuda.synchronize()
    #     m2 = torch.cuda.max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     t2 = time.time()
    #     # store results
    #     result_time = DotMap(warp=np.nan, splat=np.nan, backward=np.nan, total=(t2 - t0) * 1000)
    #     result_memory = DotMap(warp=np.nan, splat=np.nan, backward=np.nan, total=max(m1 - m0, m2 - m0))
    #     loss_val = loss.detach().clone()
    #     flow_grad = flows.grad.clone()
    #     # set grad to none
    #     flows.grad = None
    #     return result_time, result_memory, loss_val, flow_grad

    # cuda
    def cuda_once(events, flows):
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        # warp events
        warped_events = c3o.cu.iterative_3d_warp(events.view(b, -1, 5), flows, d, True, 0)
        warped_events = warped_events[..., [0, 1, 2, 4]].view(b, -1, 4)  # remove z_orig
        torch.cuda.synchronize()
        m1 = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        # splat to images
        splatted = c3o.cu.trilinear_splat(warped_events, (d + 1, h, w))
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
        "torch_naive": torch_naive_once,
        # "torch_batch": torch_batch_once,
        "cuda": cuda_once,
    }

    # benchmark
    results_time, results_memory, losses, grads = DotMap(), DotMap(), DotMap(), DotMap()
    for name, once in methods.items():
        for n, events, flows in zip(num_events, events_, flows_):
            print(f"Running {name} with {n} events")

            # max 100 events if naive torch
            if name == "torch_naive" and n > 100:
                continue

            # warmup
            once(events, flows)

            # benchmark
            for i in range(repeats):
                result_time, result_memory, loss, grad = once(events, flows)
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
