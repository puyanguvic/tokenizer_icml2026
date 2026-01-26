"""CIT Tokenizers (Controlled Interface Tokenization).

Structure
- cit_tokenizers.interface: contract + hygiene + serialization
- cit_tokenizers.cit: trainer/compiler/runtime
- cit_tokenizers.artifacts: artifact helpers
- cit_tokenizers.io: corpus/dataset IO
"""

__version__ = "0.7.0"

from .interface.contract import Contract, ContractConfig
from .artifacts.hf_artifact import save_hf_tokenizer
from .hygiene.runtime import HygieneRuntime, load_hygiene_runtime, load_tokenizer_config, assert_version_binding
from .cit.trainer import CITTrainer
from .config import CITTrainerConfig, CITBuildConfig

# Optional: `transformers` is a runtime dependency for the HF-compatible wrapper.
# Keep core (contract/compiler/runtime) importable without transformers to support
# lightweight CI and downstream usage that only needs artifacts.
try:  # pragma: no cover
    from .tokenization_cit import CITTokenizer
except Exception:  # pragma: no cover
    CITTokenizer = None  # type: ignore

__all__ = [
    "__version__",
    "CITTokenizer",
    "Contract",
    "ContractConfig",
    "save_hf_tokenizer",
    "HygieneRuntime",
    "load_hygiene_runtime",
    "load_tokenizer_config",
    "assert_version_binding",
    "CITTrainer",
    "CITTrainerConfig",
    "CITBuildConfig",
]
