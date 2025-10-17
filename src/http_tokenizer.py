import re

def http_clean_line(line: str) -> str:
    line = line.strip().replace("\r", " ").replace("\n", " ")
    line = re.sub(r"%([0-9A-Fa-f]{2})", lambda m: bytes.fromhex(m[1]).decode("latin-1"), line)
    return line

