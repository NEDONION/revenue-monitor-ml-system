def ensure_tft_available():
    try:
        from pytorch_forecasting import TemporalFusionTransformer  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 pytorch-forecasting 依赖，请先安装：pip install pytorch-forecasting lightning"
        ) from exc
