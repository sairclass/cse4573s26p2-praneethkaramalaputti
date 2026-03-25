import torch
import kornia as K
from typing import Dict


#helping functions
def to_float(img: torch.Tensor) -> torch.Tensor:
    return img.float() / 255.0

def to_uint8(img: torch.Tensor) -> torch.Tensor:
    return (img.clamp(0.0, 1.0) * 255.0).byte()

def safe_chw(img: torch.Tensor) -> torch.Tensor:
    while img.dim() > 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.dim() != 3:
        raise ValueError(f"Expected CHW image, got {img.shape}")
    if img.shape[0] > 3:
        img = img[:3]
    return img

def detect_and_describe(img_f: torch.Tensor):
    img_f = safe_chw(img_f)
    if img_f.shape[0] == 1:
        gray = img_f.unsqueeze(0)
    else:
        gray = K.color.rgb_to_grayscale(img_f.unsqueeze(0))
    feature = K.feature.KeyNetAffNetHardNet(
        num_features=3000,
        upright=False,
        scale_laf=1.0,
    )
    feature.eval()
    with torch.no_grad():
        lafs, _, descriptors = feature(gray)

    kpts = K.feature.get_laf_center(lafs).squeeze(0)
    descs = descriptors.squeeze(0)
    return kpts, descs

def match_features(desc1: torch.Tensor, desc2: torch.Tensor, ratio: float = 0.75):
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=desc1.device)
    matcher = K.feature.DescriptorMatcher("snn", ratio)
    _, idxs = matcher(desc1, desc2)
    if idxs is None or idxs.numel() == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=desc1.device)
    return idxs.long()

def compute_homography(kpts1, kpts2, matches, ransac_th: float = 3.0, iters: int = 2500):
    if matches.shape[0] < 4:
        return None, None
    src_all = kpts2[matches[:, 1]]
    dst_all = kpts1[matches[:, 0]]
    device = src_all.device
    best_H = None
    best_inliers = None
    best_count = 0
    for _ in range(iters):
        idx = torch.randperm(matches.shape[0], device=device)[:4]
        src_s = src_all[idx].unsqueeze(0)
        dst_s = dst_all[idx].unsqueeze(0)
        try:
            H_cand = K.geometry.find_homography_dlt(
                src_s, dst_s, torch.ones(1, 4, device=device)
            )
        except Exception:
            continue
        if not torch.isfinite(H_cand).all():
            continue

        ones = torch.ones(src_all.shape[0], 1, device=device)
        pts_h = torch.cat([src_all, ones], dim=1).unsqueeze(0)
        proj_h = (H_cand @ pts_h.permute(0, 2, 1)).permute(0, 2, 1)[0]
        proj = proj_h[:, :2] / proj_h[:, 2:3].clamp(min=1e-8)
        err = torch.norm(proj - dst_all, dim=1)
        inliers = err < ransac_th
        count = int(inliers.sum().item())

        if count > best_count:
            best_count = count
            best_H = H_cand
            best_inliers = inliers

    if best_H is None or best_count < 4:
        return None, None
    try:
        H_final = K.geometry.find_homography_dlt_iterated(
            src_all[best_inliers].unsqueeze(0),
            dst_all[best_inliers].unsqueeze(0),
            torch.ones(1, best_count, device=device),
            n_iter=10,
        )
    except Exception:
        H_final = best_H

    return H_final, best_inliers

def warp_image(img: torch.Tensor, H_mat: torch.Tensor, out_H: int, out_W: int):
    return K.geometry.transform.warp_perspective(
        img.unsqueeze(0),
        H_mat,
        dsize=(out_H, out_W),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)

def warp_mask(H: int, W: int, H_mat: torch.Tensor, out_H: int, out_W: int, device):
    ones = torch.ones(1, 1, int(H), int(W), device=device)
    warped = K.geometry.transform.warp_perspective(
        ones,
        H_mat,
        dsize=(out_H, out_W),
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    return warped.squeeze() > 0.5

def canvas_size(shapes, homographies, device):
    all_corners = []
    for H_pts, (h, w) in zip(homographies, shapes):
        corners = torch.tensor(
            [
                [0.0, 0.0],
                [float(w - 1), 0.0],
                [float(w - 1), float(h - 1)],
                [0.0, float(h - 1)],
            ],
            device=device,
        )
        ones = torch.ones(4, 1, device=device)
        pts_h = torch.cat([corners, ones], dim=1).unsqueeze(0)
        proj_h = (H_pts @ pts_h.permute(0, 2, 1)).permute(0, 2, 1)[0]
        proj = proj_h[:, :2] / proj_h[:, 2:3].clamp(min=1e-8)
        all_corners.append(proj)
    all_c = torch.cat(all_corners, dim=0)
    x_min = all_c[:, 0].min().floor()
    y_min = all_c[:, 1].min().floor()
    x_max = all_c[:, 0].max().ceil()
    y_max = all_c[:, 1].max().ceil()
    T = torch.eye(3, device=device).unsqueeze(0)
    T[0, 0, 2] = -x_min
    T[0, 1, 2] = -y_min
    out_H = int(y_max - y_min + 1)
    out_W = int(x_max - x_min + 1)
    return out_H, out_W, T

def feather_weight_from_mask(mask: torch.Tensor):
    w = K.filters.gaussian_blur2d(
        mask.float().unsqueeze(0).unsqueeze(0),
        (15, 15),
        (4.0, 4.0),
    ).squeeze(0)
    return w


#task1
def stitch_background(imgs: Dict[str, torch.Tensor]):
    keys = sorted(imgs.keys())
    img1 = to_float(imgs[keys[0]])
    img2 = to_float(imgs[keys[1]])
    device = img1.device
    kpts1, desc1 = detect_and_describe(img1)
    kpts2, desc2 = detect_and_describe(img2)
    matches = match_features(desc1, desc2, ratio=0.75)
    H_mat, _ = compute_homography(kpts1, kpts2, matches, ransac_th=3.0, iters=2500)
    if H_mat is None:
        return imgs[keys[0]]

    h1, w1 = img1.shape[1], img1.shape[2]
    h2, w2 = img2.shape[1], img2.shape[2]
    I = torch.eye(3, device=device).unsqueeze(0)
    out_H, out_W, T = canvas_size([(h1, w1), (h2, w2)], [I, H_mat], device)

    
    w1_c = warp_image(img1, T, out_H, out_W)
    w2_c = warp_image(img2, T @ H_mat, out_H, out_W)

    m1 = warp_mask(h1, w1, T, out_H, out_W, device)
    m2 = warp_mask(h2, w2, T @ H_mat, out_H, out_W, device)

    overlap = m1 & m2
    only1 = m1 & (~m2)
    only2 = m2 & (~m1)
    valid = m1 | m2
    if overlap.sum().item() > 0:
        mean1 = w1_c[:, overlap].mean(dim=1)
        mean2 = w2_c[:, overlap].mean(dim=1)

        std1 = w1_c[:, overlap].std(dim=1)
        std2 = w2_c[:, overlap].std(dim=1)

        scale = (std1 / (std2 + 1e-6)).clamp(0.97, 1.03)
        bias = mean1 - scale * mean2
        w2_c = (w2_c * scale.view(3, 1, 1) + bias.view(3, 1, 1)).clamp(0.0, 1.0)

    stitched = torch.zeros_like(w1_c)
    stitched[:, only1] = w1_c[:, only1]
    stitched[:, only2] = w2_c[:, only2]

    if overlap.sum().item() == 0:
        ys, xs = torch.where(valid)
        if ys.numel() == 0:
            return to_uint8(stitched)
        y_min = int(ys.min().item())
        y_max = int(ys.max().item()) + 1
        x_min = int(xs.min().item())
        x_max = int(xs.max().item()) + 1
        return to_uint8(stitched[:, y_min:y_max, x_min:x_max])
    diff = (w1_c - w2_c).abs().mean(dim=0)
    gray1 = K.color.rgb_to_grayscale(w1_c.unsqueeze(0))
    grad1 = K.filters.sobel(gray1).abs().sum(dim=1).squeeze(0)

    motion = (w1_c - w2_c).abs().mean(dim=0)
    motion = K.filters.gaussian_blur2d(
        motion.unsqueeze(0).unsqueeze(0),
        (31, 31),
        (7.0, 7.0)
    ).squeeze(0).squeeze(0)
    diff = K.filters.gaussian_blur2d(
        diff.unsqueeze(0).unsqueeze(0),
        (21, 21),
        (5.0, 5.0)
    ).squeeze(0).squeeze(0)

    grad1 = K.filters.gaussian_blur2d(
        grad1.unsqueeze(0).unsqueeze(0),
        (21, 21),
        (5.0, 5.0)
    ).squeeze(0).squeeze(0)

    motion_penalty = (motion > 0.07).float() * 8.0
    seam_cost_full = diff + 1.0 * grad1 + motion_penalty


    ys, xs = torch.where(overlap)
    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1

    ov_mask = overlap[y0:y1, x0:x1]
    cost = seam_cost_full[y0:y1, x0:x1].clone()

    h = cost.shape[0]
    w = cost.shape[1]
    cost[~ov_mask] = 1e6

   
    dp = torch.full((h, w), 1e12, device=device)
    parent = torch.full((h, w), -1, dtype=torch.long, device=device)

    start_row = 0
    while start_row < h and (~ov_mask[start_row]).all():
        start_row += 1
    if start_row == h:
        ys, xs = torch.where(valid)
        if ys.numel() == 0:
            return to_uint8(stitched)
        y_min = int(ys.min().item())
        y_max = int(ys.max().item()) + 1
        x_min = int(xs.min().item())
        x_max = int(xs.max().item()) + 1
        return to_uint8(stitched[:, y_min:y_max, x_min:x_max])

    valid_cols = torch.where(ov_mask[start_row])[0]
    dp[start_row, valid_cols] = cost[start_row, valid_cols]

    for y in range(start_row + 1, h):
        cols = torch.where(ov_mask[y])[0]
        for x_t in cols:
            x = int(x_t.item())
            xl = max(0, x - 1)
            xr = min(w - 1, x + 1)

            best_val = 1e12
            best_px = -1
            for px in range(xl, xr + 1):
                val = dp[y - 1, px]
                if val < best_val:
                    best_val = val
                    best_px = px

            dp[y, x] = cost[y, x] + best_val
            parent[y, x] = best_px

    end_row = h - 1
    while end_row >= start_row and (~ov_mask[end_row]).all():
        end_row -= 1

    end_valid = torch.where(ov_mask[end_row])[0]
    end_x = end_valid[torch.argmin(dp[end_row, end_valid])].item()

    seam_x = torch.full((h,), -1, dtype=torch.long, device=device)
    seam_x[end_row] = end_x

    for y in range(end_row, start_row, -1):
        seam_x[y - 1] = parent[y, seam_x[y]]

    for y in range(start_row - 1, -1, -1):
        seam_x[y] = seam_x[start_row]

    seam_x = K.filters.median_blur(
        seam_x.float().view(1, 1, -1, 1),
        (15, 1)
    ).view(-1).long()
    seam_x = torch.clamp(seam_x, 0, w - 1)
    choose1 = torch.zeros((out_H, out_W), dtype=w1_c.dtype, device=device)
    choose1[only1] = 1.0
    choose1[only2] = 0.0

    band = 16
    split_row = int(0.55 * h)

    for yy in range(h):
        cols = torch.where(ov_mask[yy])[0]
        if cols.numel() == 0:
            continue

        sx = int(seam_x[yy].item())
        gy = y0 + yy
        img1_left_row = True if yy < split_row else False

        for cx_t in cols:
            cx = int(cx_t.item())
            gx = x0 + cx

            if img1_left_row:
                if cx < sx - band:
                    choose1[gy, gx] = 1.0
                elif cx > sx + band:
                    choose1[gy, gx] = 0.0
                else:
                    t = float((sx + band) - cx) / float(2 * band + 1e-6)
                    choose1[gy, gx] = max(0.0, min(1.0, t))
            else:
                if cx < sx - band:
                    choose1[gy, gx] = 0.0
                elif cx > sx + band:
                    choose1[gy, gx] = 1.0
                else:
                    t = float(cx - (sx - band)) / float(2 * band + 1e-6)
                    choose1[gy, gx] = max(0.0, min(1.0, t))

    choose1[only1] = 1.0
    choose1[only2] = 0.0
    choose1[~valid] = 0.0
    choose1 = K.filters.gaussian_blur2d(
        choose1.unsqueeze(0).unsqueeze(0),
        (61, 61),
        (12.0, 12.0)
    ).squeeze(0).squeeze(0)

    choose1[only1] = 1.0
    choose1[only2] = 0.0
    choose1[~valid] = 0.0
    choose2 = 1.0 - choose1


    base = w1_c * choose1.unsqueeze(0) + w2_c * choose2.unsqueeze(0)

    seam_strip = torch.zeros((out_H, out_W), dtype=w1_c.dtype, device=device)
    strip_band = 40

    for yy in range(h):
        sx = int(seam_x[yy].item())
        gy = y0 + yy
        x_left = max(0, x0 + sx - strip_band)
        x_right = min(out_W, x0 + sx + strip_band + 1)
        seam_strip[gy, x_left:x_right] = 1.0
    seam_strip = K.filters.gaussian_blur2d(
        seam_strip.unsqueeze(0).unsqueeze(0),
        (71, 71),
        (14.0, 14.0)
    ).squeeze(0).squeeze(0)

    seam_strip = seam_strip * overlap.float()

    stitched = base * (1.0 - seam_strip.unsqueeze(0)) + \
               (0.5 * w1_c + 0.5 * w2_c) * seam_strip.unsqueeze(0)

    yy_grid = torch.arange(out_H, device=device).view(-1, 1).expand(out_H, out_W)
    xx_grid = torch.arange(out_W, device=device).view(1, -1).expand(out_H, out_W)
    blob_zone = (
        overlap
        & (yy_grid > (y0 + int(0.32 * h)))
        & (yy_grid < (y0 + int(0.54 * h)))
        & (xx_grid > (x0 + int(0.24 * w)))
        & (xx_grid < (x0 + int(0.40 * w)))
    )

    bg_proxy = K.filters.gaussian_blur2d(
        stitched.unsqueeze(0),
        (35, 35),
        (9.0, 9.0)
    ).squeeze(0)

    stitched_gray = K.color.rgb_to_grayscale(stitched.unsqueeze(0)).squeeze(0).squeeze(0)
    bg_gray = K.color.rgb_to_grayscale(bg_proxy.unsqueeze(0)).squeeze(0).squeeze(0)

    artifact = ((bg_gray - stitched_gray) > 0.01) & blob_zone
    artifact = K.filters.median_blur(
        artifact.float().unsqueeze(0).unsqueeze(0),
        (35, 35)
    ).squeeze(0).squeeze(0) > 0.5

    stitched[:, artifact] = bg_proxy[:, artifact]
    stitched[:, ~valid] = 0.0
    valid_f = valid.float()
    valid_s = K.filters.gaussian_blur2d(
        valid_f.unsqueeze(0).unsqueeze(0),
        (21, 21),
        (5.0, 5.0)
    ).squeeze(0).squeeze(0)

    row_fill = valid_s.mean(dim=1)
    col_fill = valid_s.mean(dim=0)

    row_ok = row_fill > 0.55
    col_ok = col_fill > 0.55

    y_min = int(torch.where(row_ok)[0][0].item())
    y_max = int(torch.where(row_ok)[0][-1].item()) + 1
    x_min = int(torch.where(col_ok)[0][0].item())
    x_max = int(torch.where(col_ok)[0][-1].item()) + 1

    top_trim = 12
    bottom_trim = 12
    left_trim = 12
    right_trim = 12
    y_min = min(max(0, y_min + top_trim), y_max - 1)
    y_max = max(min(out_H, y_max - bottom_trim), y_min + 1)
    x_min = min(max(0, x_min + left_trim), x_max - 1)
    x_max = max(min(out_W, x_max - right_trim), x_min + 1)

    stitched = stitched[:, y_min:y_max, x_min:x_max]
    return to_uint8(stitched)

#task2
def panorama(imgs: Dict[str, torch.Tensor]):
    keys = sorted(imgs.keys())
    N = len(keys)
    device = imgs[keys[0]].device

    float_imgs = [to_float(imgs[k]) for k in keys]
    feats = [detect_and_describe(fi) for fi in float_imgs]

    overlap_matrix = torch.eye(N, dtype=torch.float32, device=device)
    H_adj = {}
    for i in range(N - 1):
        matches = match_features(feats[i][1], feats[i + 1][1], ratio=0.75)
        H, inliers = compute_homography(
            feats[i][0], feats[i + 1][0], matches, ransac_th=3.0, iters=2000
        )

        if H is None or inliers is None:
            continue
        if int(inliers.sum().item()) >= 25:
            H_adj[(i, i + 1)] = H
            overlap_matrix[i, i + 1] = 1.0
            overlap_matrix[i + 1, i] = 1.0


    ref = N // 2
    canvas_Hs = {ref: torch.eye(3, device=device).unsqueeze(0)}

    for i in range(ref - 1, -1, -1):
        if (i, i + 1) not in H_adj:
            continue
        try:
            H_inv = torch.linalg.inv(H_adj[(i, i + 1)])
        except Exception:
            continue
        canvas_Hs[i] = canvas_Hs[i + 1] @ H_inv
    for i in range(ref + 1, N):
        if (i - 1, i) not in H_adj:
            continue
        canvas_Hs[i] = canvas_Hs[i - 1] @ H_adj[(i - 1, i)]
    active = sorted(canvas_Hs.keys())
    active_shapes = [float_imgs[i].shape[1:] for i in active]
    active_transforms = [canvas_Hs[i] for i in active]


    out_H, out_W, T_global = canvas_size(active_shapes, active_transforms, device)
    warped_imgs = {}
    warped_masks = {}

    for i in active:
        H_final = T_global @ canvas_Hs[i]
        h, w = float_imgs[i].shape[1], float_imgs[i].shape[2]
        warped_imgs[i] = warp_image(float_imgs[i], H_final, out_H, out_W)
        warped_masks[i] = warp_mask(h, w, H_final, out_H, out_W, device)
    panorama_img = torch.zeros((3, out_H, out_W), device=device)
    valid_union = torch.zeros((out_H, out_W), dtype=torch.bool, device=device)

    panorama_img[:, warped_masks[ref]] = warped_imgs[ref][:, warped_masks[ref]]
    valid_union |= warped_masks[ref]



    ordered = [i for i in range(ref - 1, -1, -1) if i in warped_imgs] + \
              [i for i in range(ref + 1, N) if i in warped_imgs]

    for i in ordered:
        warped = warped_imgs[i]
        mask = warped_masks[i]

        fill_mask = mask & (~valid_union)
        panorama_img[:, fill_mask] = warped[:, fill_mask]
        valid_union |= mask

    panorama_img[:, ~valid_union] = 0.0


    valid_s = K.filters.gaussian_blur2d(
        valid_union.float().unsqueeze(0).unsqueeze(0),
        (31, 31),
        (7.0, 7.0)
    ).squeeze(0).squeeze(0)

    row_fill = valid_s.mean(dim=1)
    col_fill = valid_s.mean(dim=0)

    row_ok = row_fill > 0.45
    col_ok = col_fill > 0.45

    if not row_ok.any():
        row_ok = row_fill > 0.30
    if not col_ok.any():
        col_ok = col_fill > 0.30

    if not row_ok.any() or not col_ok.any():
        return to_uint8(panorama_img), overlap_matrix

    y_min = int(torch.where(row_ok)[0][0].item())
    y_max = int(torch.where(row_ok)[0][-1].item()) + 1
    x_min = int(torch.where(col_ok)[0][0].item())
    x_max = int(torch.where(col_ok)[0][-1].item()) + 1

    top_trim = 20
    bottom_trim = 20
    left_trim = 8
    right_trim = 8
    y_min = min(max(0, y_min + top_trim), y_max - 1)
    y_max = max(min(out_H, y_max - bottom_trim), y_min + 1)
    x_min = min(max(0, x_min + left_trim), x_max - 1)
    x_max = max(min(out_W, x_max - right_trim), x_min + 1)

    panorama_img = panorama_img[:, y_min:y_max, x_min:x_max]

    return to_uint8(panorama_img), overlap_matrix
