"""USD cost from logged usage. Prices per MTok (platform.claude.com/docs pricing, 2026-11-03)."""

# base input, 5m cache write, cache read, output
PRICES = {
    "claude-haiku-5-5":       (0.10, 0.125, 0.01, 0.50),   # prompt <= 100K tokens
    "claude-haiku-5-5@>100k": (0.50, 0.625, 0.05, 2.50),   # prompt  > 100K tokens
    "claude-sonnet-5-5":      (2.00, 2.50, 0.10, 10.0),
    "claude-opus-5-5":        (4.00, 5.00, 0.20, 20.0),
    "claude-fable-5-1":       (10.0, 12.5, 0.25, 50.0),   # cache read 2.5% of base
}


def record_cost(r: dict) -> float:
    model = r["model"]
    if model.startswith("mock"):
        return 0.0
    total_in = r.get("total_input_tokens", r.get("input_tokens", 0))
    if model == "claude-haiku-5-5" and total_in > 100_000:
        model = "claude-haiku-5-5@>100k"
    base, write, read, out = PRICES[model]
    return (r.get("input_tokens", 0) * base
            + r.get("cache_creation_input_tokens", 0) * write
            + r.get("cache_read_input_tokens", 0) * read
            + r.get("output_tokens", 0) * out) / 1e6
