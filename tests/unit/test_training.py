"""
Training building blocks: fast checks with no GPU, no dataset and no downloads.

They cover the parts most likely to break silently: the JPEG augmentation, the class
order, the training plan, early stopping, config validation and which layers train.
"""

import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image
from pydantic import ValidationError

from scripts.train_config import TrainConfig
from scripts.training.data import RandomJPEG, get_dataloaders, get_transforms
from scripts.training.model import build_model, set_backbone_trainable
from scripts.training.phases import EarlyStopping, EvalResult, training_plan


def _noise_image(size=(64, 48), mode="RGB") -> Image.Image:
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 256, (size[1], size[0], 3), generator=g, dtype=torch.uint8)
    return Image.fromarray(data.numpy()).convert(mode)


# -- RandomJPEG ---------------------------------------------------------------


def test_jpeg_with_p0_returns_the_image_untouched() -> None:
    img = _noise_image()
    assert RandomJPEG(p=0.0, quality=(60, 95))(img) is img


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_jpeg_with_p1_keeps_size_and_gives_rgb(mode: str) -> None:
    img = _noise_image(mode=mode)
    out = RandomJPEG(p=1.0, quality=(60, 60))(img)
    assert out.size == img.size
    assert out.mode == "RGB"


def test_jpeg_actually_changes_the_pixels() -> None:
    img = _noise_image()
    out = RandomJPEG(p=1.0, quality=(10, 10))(img)
    assert out.tobytes() != img.tobytes()


def test_jpeg_can_be_pickled_for_dataloader_workers() -> None:
    restored = pickle.loads(pickle.dumps(RandomJPEG(p=0.5, quality=(60, 95))))
    assert (restored.p, restored.quality) == (0.5, (60, 95))


def test_jpeg_is_only_in_the_training_transform() -> None:
    train_tf, val_tf = get_transforms(TrainConfig(jpeg_prob=0.3, jpeg_quality=(70, 90)))
    first = train_tf.transforms[0]
    assert isinstance(first, RandomJPEG)
    assert (first.p, first.quality) == (0.3, (70, 90))
    assert not any(isinstance(t, RandomJPEG) for t in val_tf.transforms)


# -- Dataset checks ------------------------------------------------------------


def test_missing_dataset_folder_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        get_dataloaders(TrainConfig(dataset_dir=tmp_path / "nowhere"))


def test_wrong_class_folders_are_refused(tmp_path) -> None:
    for split in ("train", "val"):
        for label in ("FAKE", "REAL"):  # inference expects AI_GENERATED, REAL
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            _noise_image().save(folder / "a.png")
    with pytest.raises(ValueError, match="Expected"):
        get_dataloaders(TrainConfig(dataset_dir=tmp_path, num_workers=1))


# -- Training plan and early stopping -----------------------------------------


def test_plan_has_a_head_phase_then_a_finetune_phase() -> None:
    config = TrainConfig(
        epochs=7, learning_rate=1e-3, finetune_epochs=4, finetune_lr=1e-4
    )
    head, finetune = training_plan(config)
    assert (head.number, head.epochs, head.lr, head.train_backbone) == (
        1,
        7,
        1e-3,
        False,
    )
    assert (finetune.number, finetune.epochs, finetune.lr, finetune.train_backbone) == (
        2,
        4,
        1e-4,
        True,
    )


def test_unfrozen_config_trains_the_backbone_from_the_start() -> None:
    head, _ = training_plan(TrainConfig(freeze_backbone=False))
    assert head.train_backbone


def test_early_stopping_waits_exactly_patience_epochs() -> None:
    stopper = EarlyStopping(patience=2)
    assert not stopper.step(1.0)  # first value is an improvement
    assert not stopper.step(1.1)  # 1 bad epoch
    assert stopper.step(1.2)  # 2 bad epochs: stop


def test_an_improvement_resets_early_stopping() -> None:
    stopper = EarlyStopping(patience=2)
    stopper.step(1.0)
    stopper.step(1.1)
    assert not stopper.step(0.9)  # improvement resets the count
    assert not stopper.step(1.0)


def test_eval_result_rounds_every_score() -> None:
    result = EvalResult(
        loss=0.123456, accuracy=0.9, precision=0.8, recall=0.7, f1_score=0.654321
    )
    assert result.rounded() == {
        "loss": 0.1235,
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.7,
        "f1_score": 0.6543,
    }


# -- Config --------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"jpeg_prob": 1.5},
        {"jpeg_prob": -0.1},
        {"jpeg_quality": (95, 60)},
        {"jpeg_quality": (0, 90)},
        {"finetune_lr": 0},
        {"finetune_epochs": 0},
    ],
)
def test_config_refuses_bad_values(bad: dict) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(**bad)


def test_every_phase_setting_is_logged() -> None:
    params = TrainConfig().to_mlflow_params()
    for key in (
        "phase1_epochs",
        "phase1_lr",
        "phase2_epochs",
        "phase2_lr",
        "jpeg_prob",
    ):
        assert key in params


# -- Model ---------------------------------------------------------------------


def _trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_frozen_model_trains_only_the_head() -> None:
    model = build_model(2, freeze_backbone=True, pretrained=False)
    assert _trainable(model.features) == 0
    assert _trainable(model.classifier) > 0


def test_unfreezing_makes_the_whole_network_trainable() -> None:
    model = build_model(2, freeze_backbone=True, pretrained=False)
    set_backbone_trainable(model, True)
    assert _trainable(model) == sum(p.numel() for p in model.parameters())


def test_model_outputs_one_score_per_class() -> None:
    model = build_model(2, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.zeros(3, 3, 224, 224)).shape == (3, 2)


# -- Trainer -------------------------------------------------------------------


def _tiny_loader():
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(0)
    images = torch.randn(8, 3, 64, 64, generator=g)
    labels = torch.tensor([0, 1] * 4)
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_a_phase_saves_and_restores_the_best_weights(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from scripts.training.phases import Phase
    from scripts.training.trainer import Trainer

    model = build_model(2, freeze_backbone=True, pretrained=False)
    checkpoint = tmp_path / "best.pt"
    trainer = Trainer(model, torch.device("cpu"), checkpoint)
    loader = _tiny_loader()

    result = trainer.run_phase(Phase(1, "head", 2, 1e-3, 5, False), loader, loader)

    assert result.epochs_run == 2
    assert result.best is not None and result.best == trainer.best
    saved = torch.load(checkpoint, weights_only=True)
    assert set(saved) == {
        "model_state_dict",
        "metrics",
    }  # weights only, no optimizer state
    for name, tensor in model.state_dict().items():
        assert torch.equal(
            tensor, saved["model_state_dict"][name]
        )  # model holds the best weights


def test_a_worse_phase_does_not_replace_the_best_model(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from dataclasses import replace

    from scripts.training.phases import Phase
    from scripts.training.trainer import Trainer

    model = build_model(2, freeze_backbone=True, pretrained=False)
    checkpoint = tmp_path / "best.pt"
    trainer = Trainer(model, torch.device("cpu"), checkpoint)
    loader = _tiny_loader()
    trainer.run_phase(Phase(1, "head", 1, 1e-3, 5, False), loader, loader)

    trainer.best = replace(trainer.best, loss=-1.0)  # make phase 1 unbeatable
    saved_before = checkpoint.read_bytes()
    trainer.run_phase(Phase(2, "finetune", 1, 1e-3, 3, True), loader, loader)

    assert trainer.best_phase.number == 1
    assert checkpoint.read_bytes() == saved_before  # phase 2 never overwrote it
    saved = torch.load(checkpoint, weights_only=True)["model_state_dict"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(
            tensor, saved[name]
        )  # the model went back to phase 1's weights
