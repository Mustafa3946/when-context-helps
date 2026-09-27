"""Deprecated: use ground_truth_streaming.py (DuckDB-backed)."""

def main(*_args, **_kwargs):
    raise RuntimeError(
        "build_ground_truth_bgl.py is deprecated. Use: python -m src.evaluation.ground_truth_streaming"
    )

if __name__ == "__main__":
    main()
