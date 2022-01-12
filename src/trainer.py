import segmentation_models_pytorch as smp
import torch.nn as nn
import torchmetrics
from torchgeo.models.fcn import FCN
from torchgeo.trainers import SemanticSegmentationTask


class FocalDiceLoss(nn.Module):
    def __init__(
        self, mode: str = "multiclass", ignore_index: int = 0, normalized: bool = False
    ):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(
            mode=mode, ignore_index=ignore_index, normalized=normalized
        )
        self.dice_loss = smp.losses.DiceLoss(mode=mode, ignore_index=ignore_index)

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets) + self.dice_loss(preds, targets)


class DFC2022SemanticSegmentationTask(SemanticSegmentationTask):
    def config_task(self):
        if self.hparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "fcn":
            self.model = FCN(
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
                num_filters=self.hparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

        if self.hparams["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss(  # type: ignore[attr-defined]
                ignore_index=-1000 if self.ignore_zeros is None else 0
            )
        elif self.hparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif self.hparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=self.ignore_zeros, normalized=True
            )
        elif self.hparams["loss"] == "focaldice":
            self.loss = FocalDiceLoss(
                mode="multiclass",
                ignore_index=0 if self.ignore_zeros else None,
                normalized=True,
            )
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "OverallAccuracy": torchmetrics.Accuracy(
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "OverallPrecision": torchmetrics.Precision(
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "OverallRecall": torchmetrics.Recall(
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "AverageAccuracy": torchmetrics.Accuracy(
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "AveragePrecision": torchmetrics.Precision(
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "AverageRecall": torchmetrics.Recall(
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "IoU": torchmetrics.IoU(
                    num_classes=self.hparams["num_classes"],
                    ignore_index=self.ignore_zeros,
                ),
                "F1Score": torchmetrics.FBeta(
                    num_classes=self.hparams["num_classes"],
                    beta=1.0,
                    average="micro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
