from dataclasses import dataclass


@dataclass
class PromptTemplateConfig:
    user_prefix: str = "USER:"
    assistant_prefix: str = "ASSISTANT:"
    question: str = "这张医学影像显示了什么？"


def build_classification_dialog(label: str, config: PromptTemplateConfig | None = None) -> str:
    """Convert image-label sample into an instruction-style VLM dialog."""
    config = config or PromptTemplateConfig()
    return (
        f"{config.user_prefix} <image>\\n"
        f"{config.question}\\n"
        f"{config.assistant_prefix} {label}"
    )
