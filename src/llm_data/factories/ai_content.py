import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol

from daft import DataFrame, col

ScoreNormalization = Literal["softmax", "sigmoid"]
AILabel = str | int


class SequenceClassificationPredictor(Protocol):
    @property
    def id2label(self) -> Mapping[int | str, str]: ...

    def __call__(self, texts: list[str]) -> Sequence[Sequence[float]]: ...


def _resolve_label_index(
    ai_label: AILabel,
    id2label: Mapping[int | str, str],
    num_labels: int,
) -> int:
    if isinstance(ai_label, bool):
        raise TypeError("ai_label must be a label name or integer index")

    if isinstance(ai_label, int):
        if not 0 <= ai_label < num_labels:
            raise ValueError(
                f"ai_label index {ai_label} is outside the model's {num_labels} labels"
            )
        return ai_label

    normalized_labels = {int(index): label for index, label in id2label.items()}
    exact_matches = [
        index for index, label in normalized_labels.items() if label == ai_label
    ]
    if not exact_matches:
        exact_matches = [
            index
            for index, label in normalized_labels.items()
            if label.casefold() == ai_label.casefold()
        ]
    if len(exact_matches) != 1:
        available = ", ".join(
            f"{index}={label!r}" for index, label in sorted(normalized_labels.items())
        )
        raise ValueError(
            f"ai_label {ai_label!r} did not identify exactly one output label; "
            f"available labels: {available or 'none'}"
        )

    label_index = exact_matches[0]
    if not 0 <= label_index < num_labels:
        raise ValueError(
            f"label mapping index {label_index} is outside the model's "
            f"{num_labels} outputs"
        )
    return label_index


def normalize_label_scores(
    logits: Sequence[Sequence[float]],
    *,
    ai_label: AILabel,
    id2label: Mapping[int | str, str],
    normalization: ScoreNormalization = "softmax",
) -> list[float]:
    """Convert classifier logits into scores for the configured AI label."""
    if normalization not in {"softmax", "sigmoid"}:
        raise ValueError("normalization must be 'softmax' or 'sigmoid'")
    if not logits:
        return []

    num_labels = len(logits[0])
    if num_labels == 0:
        raise ValueError("classifier returned no output labels")
    label_index = _resolve_label_index(ai_label, id2label, num_labels)

    scores = []
    for row in logits:
        if len(row) != num_labels:
            raise ValueError("classifier returned inconsistent output widths")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("classifier returned a non-finite logit")

        if normalization == "softmax":
            maximum = max(values)
            exponentials = [math.exp(value - maximum) for value in values]
            score = exponentials[label_index] / sum(exponentials)
        else:
            value = values[label_index]
            if value >= 0:
                score = 1.0 / (1.0 + math.exp(-value))
            else:
                exponential = math.exp(value)
                score = exponential / (1.0 + exponential)
        scores.append(score)

    return scores


def score_text_batch(
    texts: Sequence[str | None],
    predictor: SequenceClassificationPredictor,
    *,
    ai_label: AILabel,
    normalization: ScoreNormalization = "softmax",
    empty_score: float = 0.0,
) -> list[float]:
    """Score non-empty texts while assigning a deterministic score to empty inputs."""
    if not 0.0 <= empty_score <= 1.0:
        raise ValueError("empty_score must be between 0 and 1")

    scores = [float(empty_score)] * len(texts)
    valid_indices = [
        index for index, text in enumerate(texts) if text is not None and text.strip()
    ]
    if not valid_indices:
        return scores

    valid_texts = [text for text in texts if text is not None and text.strip()]
    logits = predictor(valid_texts)
    valid_scores = normalize_label_scores(
        logits,
        ai_label=ai_label,
        id2label=predictor.id2label,
        normalization=normalization,
    )
    if len(valid_scores) != len(valid_indices):
        raise ValueError(
            "classifier returned a different number of outputs than input texts"
        )
    for index, score in zip(valid_indices, valid_scores, strict=True):
        scores[index] = score
    return scores


class _HuggingFaceSequenceClassificationPredictor:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_length: int | None,
        device: str,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self._model.eval()
        self._model.to(device)
        self._device = device
        self._max_length = max_length
        self._id2label = dict(self._model.config.id2label)

    @property
    def id2label(self) -> Mapping[int | str, str]:
        return self._id2label

    def __call__(self, texts: list[str]) -> Sequence[Sequence[float]]:
        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self._max_length is not None:
            tokenizer_kwargs["max_length"] = self._max_length
        inputs = self._tokenizer(texts, **tokenizer_kwargs)
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        return self._model(**inputs).logits.detach().cpu().tolist()


def ai_content_scorer_factory(
    *,
    model_name_or_path: str,
    ai_label: AILabel,
    batch_size: int = 16,
    max_length: int | None = None,
    normalization: ScoreNormalization = "softmax",
    empty_score: float = 0.0,
    gpus: float = 0,
    cpus: float | None = None,
    predictor_factory: Callable[[], SequenceClassificationPredictor] | None = None,
):
    import daft
    from daft import DataType, Series

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be at least 1")
    if normalization not in {"softmax", "sigmoid"}:
        raise ValueError("normalization must be 'softmax' or 'sigmoid'")
    if not 0.0 <= empty_score <= 1.0:
        raise ValueError("empty_score must be between 0 and 1")

    @daft.cls(gpus=gpus, cpus=cpus)
    class HuggingFaceAIContentScorer:
        def __init__(self):
            self.predictor = None

        def lazy_load(self) -> SequenceClassificationPredictor:
            if self.predictor is None:
                if predictor_factory is not None:
                    self.predictor = predictor_factory()
                else:
                    device = "cuda" if gpus > 0 else "cpu"
                    self.predictor = _HuggingFaceSequenceClassificationPredictor(
                        model_name_or_path,
                        max_length=max_length,
                        device=device,
                    )
            return self.predictor

        @daft.method.batch(return_dtype=DataType.float64(), batch_size=batch_size)
        def score_batch(self, text: Series) -> list[float]:
            texts = text.to_pylist()
            if all(value is None or not value.strip() for value in texts):
                return [float(empty_score)] * len(texts)
            return score_text_batch(
                texts,
                self.lazy_load(),
                ai_label=ai_label,
                normalization=normalization,
                empty_score=empty_score,
            )

    return HuggingFaceAIContentScorer().score_batch


class AIContentScorer:
    """Add a model-specific AI-content score without filtering input rows."""

    def __init__(
        self,
        model_name_or_path: str,
        ai_label: AILabel,
        input_column: str = "text",
        output_column: str = "ai_content_score",
        batch_size: int = 16,
        max_length: int | None = None,
        normalization: ScoreNormalization = "softmax",
        empty_score: float = 0.0,
        gpus: float = 0,
        cpus: float | None = None,
        predictor_factory: Callable[[], SequenceClassificationPredictor] | None = None,
        name: str = "AIContentScorer",
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.name = name
        self.score_batch = ai_content_scorer_factory(
            model_name_or_path=model_name_or_path,
            ai_label=ai_label,
            batch_size=batch_size,
            max_length=max_length,
            normalization=normalization,
            empty_score=empty_score,
            gpus=gpus,
            cpus=cpus,
            predictor_factory=predictor_factory,
        )

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column,
            self.score_batch(col(self.input_column)),
        )
