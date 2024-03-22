import importlib
import os

for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        criterion_name = file[: file.find(".py")]
        importlib.import_module(
            "examples.speechllm.criterions." + criterion_name
        )
