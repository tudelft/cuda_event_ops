import torch
import torch.nn.functional as F


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
    events[..., 0:2] += (t_ref - events[..., 2:3]) * flow
    if not keep_ts:
        events[..., 2:3] = t_ref
    return events


def get_event_flow_3d(events, flow):
    """
    Bilinearly samples flow at event locations.

    Args:
        events (torch.Tensor): A tensor of shape (b, d, n, 5), where each event has (x, y, z, zi, val).
        flow (torch.Tensor): A tensor of shape (b, d, h, w, 2), where each flow has (u, v).

    Returns:
        torch.Tensor: A tensor of shape (b, d, n, 2) of (u, v) flow for each event
    """

    # discretize ts (no bilinear in time)
    # grid: order (x, y, t) in last dim and normalize to (-1, 1)
    _, d, h, w, _ = flow.shape
    x, y = events[..., 0:2].unbind(-1)
    z = torch.ones_like(x) * torch.arange(d, dtype=events.dtype, device=events.device).view(-1, 1)
    resolution = torch.tensor([w, h, d], dtype=events.dtype, device=events.device)  # needs clamp if d == 1
    grid_norm = 2 * torch.stack([x, y, z], dim=-1).unsqueeze(3) / (resolution - 1).clamp(min=1) - 1

    # use grid_sample so idx can be float
    # align corners because warping to pixel centers
    # flow shape: (b, d, h, w, 2) -> (b, 2, d, n, w); idx_norm shape: (b, d, n, 1, 3)
    # returns (b, 2, d, n, 1)
    flow = F.grid_sample(flow.permute(0, 4, 1, 2, 3), grid_norm, mode="bilinear", align_corners=True)

    return flow.squeeze(-1).permute(0, 2, 3, 1)


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
    mask = (idx >= 0) & (idx <= resolution - 1)
    return mask.all(dim=-1, keepdim=True)


def iterative_3d_warp(events, flows, num_warps):
    """
    Iteratively warps events in 3D using bilinearly-sampled flows.

    Args:
        events (torch.Tensor): A tensor of shape (b, n, 5), where each event has (x, y, z, zi, val).
        flows (torch.Tensor): A tensor of shape (b, d, h, w, 2), where each flow has (u, v).
        num_warps (int): The number of warping stages.

    Returns:
        torch.Tensor: A tensor of shape (b, n, d + 1, 5), where each event has (x, y, z, z_orig, val).
    """

    # get deblurring window and resolution
    b, d, h, w, _ = flows.shape
    resolution = torch.tensor([w, h], device=flows.device)

    # repeat for simultaneous fw and bw warping
    events = events[..., [0, 1, 2, 4]].view(b, d, -1, 4).contiguous()
    events = events.repeat(1, 2, 1, 1)  # copies

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
        select_events = events[:, t_events]
        select_flow = flows[:, t_flow]

        # sample flow that will warp events at event locations
        # ensure integer t because we don't want bilinear there
        # (is a concat of fw/bw, not chronological)
        select_event_flow = get_event_flow_3d(select_events, select_flow)

        # fw/bw warp events to reference time
        # overwrite ts with reference time
        # all in-place
        select_events = linear_warp(select_events, select_event_flow, t_ref.view(1, -1, 1, 1), keep_ts=False)

        # discard events warped outside image
        mask_inside = compute_inside_mask(select_events[..., 0:2], resolution)
        select_events[..., [0, 1, 3]] *= mask_inside  # mask idx and p, can be done in-place

        # save warped events, no copy here
        events[:, t_events] = select_events

        # save warped events, ts and inside mask for border compensation
        # also take into account mask outside base
        for j, (src, dst) in enumerate(zip([*t0, *t1], t_ref)):
            warped_events[dst][src] = select_events[:, j : j + 1] if i < num_warps else None
            warped_mask_inside[src] = (
                warped_mask_inside[src] * mask_inside[:, j : j + 1]
                if warped_mask_inside[src] is not None
                else mask_inside[:, j : j + 1]
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
    left_x = idx[..., 0].floor()
    right_x = (idx[..., 0] + 1).floor()
    top_y = idx[..., 1].floor()
    bot_y = (idx[..., 1] + 1).floor()

    # combine to 2d corners
    top_left = torch.stack([left_x, top_y], dim=-1)
    top_right = torch.stack([right_x, top_y], dim=-1)
    bottom_left = torch.stack([left_x, bot_y], dim=-1)
    bottom_right = torch.stack([right_x, bot_y], dim=-1)

    # corners sequentially in event dimension
    idx_round = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=-2)
    idx = idx.repeat(1, 4, 1) if idx.dim() == 3 else idx.repeat(1, 1, 4, 1)

    # compute inverse l1 distances
    # event_flow uses max(0, 1 - abs(idx - idx_round)), but that kills gradients at bounds
    # clamp doesn't: https://github.com/pytorch/pytorch/issues/7002, should be better
    inv_dist = (1 - (idx - idx_round).abs()).clamp(min=0)

    # weight = product of distances
    weight = inv_dist.prod(dim=-1, keepdim=True)

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
    image = torch.zeros(*idx.shape[:-2], resolution.prod(), 1, dtype=idx.dtype, device=idx.device)

    # convert idx to 1d
    idx = (idx[..., 1:2] * resolution[1] + idx[..., 0:1]).long()

    # accumulate and reshape to 2d
    image.scatter_add_(-2, idx, weight * mask)
    # image = scatter_add(weight * mask, idx, dim=-1)
    return image.unflatten(-2, resolution.tolist()).permute(0, 1, 4, 2, 3)


def trilinear_splat(events, grid_resolution):
    # get deblurring window and resolution
    _, _, b, _, _ = events.shape
    d, h, w = grid_resolution
    resolution = torch.tensor([h, w], device=events.device)

    # iwe and iwt at each reference time
    iwe = torch.zeros(d, b, 1, h, w, dtype=events.dtype, device=events.device)

    # build iwe and iwt
    for i in range(d):
        # inverse product of l1 distances to closest pixels
        corners, weights = inv_l1_dist_prod(events[i][..., 0:2])

        # discard corners outside image
        mask_inside = compute_inside_mask(corners, resolution)
        corners = corners * mask_inside
        weights = weights * mask_inside

        # iwe
        values = events[i][..., 3:4].repeat(1, 1, 4, 1)  # repeat for corners
        iwe[i] += accumulate_to_image(corners, weights * values, torch.ones_like(weights), resolution).sum(0)

    # NOTE: temporary
    iwe = iwe.squeeze(2).permute(1, 0, 2, 3).contiguous()

    return iwe
