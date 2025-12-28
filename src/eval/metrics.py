import torch


def eval_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    # 计算常用回归指标。
    abs_diff = torch.abs(preds - targets)
    mae = torch.mean(abs_diff).item()
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    eps = 1e-8
    mape = torch.mean(abs_diff / (torch.abs(targets) + eps)).item()
    smape = torch.mean(2 * abs_diff / (torch.abs(preds) + torch.abs(targets) + eps)).item()

    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = (1 - ss_res / (ss_tot + eps)).item()

    p50 = torch.quantile(abs_diff, 0.5).item()
    p90 = torch.quantile(abs_diff, 0.9).item()

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "p50_err": p50,
        "p90_err": p90,
    }
