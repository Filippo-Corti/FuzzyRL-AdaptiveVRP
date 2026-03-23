def to_bin(x: float) -> str:
    if x < 0.25:
        return "LOW"
    elif x < 0.75:
        return "MEDIUM"
    else:
        return "HIGH"
