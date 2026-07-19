"""النموذج التأسيسي ذاتي الإشراف (Self-Supervised Foundation Model) — المحطة 4.

تُبنى هذه المحطة على مخرجات ``nq.features`` (Feature Store) عبر بنية تحتية للتدريب
خالية من التسريب الزمني، ومشفّر تأسيسي أساسي (baseline) قابل للاستبدال:

* ``splitting``    — تقسيم زمني walk-forward مع purge/embargo.
* ``windowing``    — تقطيع تسلسلات سببية متعدّدة المقاييس (SequenceDataset).
* ``preprocessing``— تطبيع سببي (fit-on-train ثم تطبيق للأمام).
* ``encoder``      — تعلّم تمثيلي ذاتي الإشراف (PCAEncoder) خلف ``Encoder`` Protocol.
* ``masking``      — النمذجة المُقنّعة (masked event/state reconstruction).
* ``world_model``  — نموذج العالم التنبّئي (next-state prediction).
* ``contrastive``  — التعلّم التبايني (augmentations + InfoNCE).

كل الواجهات مصمّمة ليُوصَل خلفها لاحقًا مشفّر عصبي (Transformer) دون تغيير المسار
أو ضمانات منع التسريب.
"""

from __future__ import annotations

from nq.models.contrastive import augment_windows, info_nce_loss
from nq.models.encoder import Encoder, PCAEncoder
from nq.models.masking import mask_matrix, masked_reconstruction_error
from nq.models.preprocessing import CausalStandardScaler
from nq.models.splitting import WalkForwardFold, purged_walk_forward_split
from nq.models.windowing import SequenceDataset, build_sequences
from nq.models.world_model import NextStatePredictor, r2_score

__all__ = [
    "CausalStandardScaler",
    "Encoder",
    "NextStatePredictor",
    "PCAEncoder",
    "SequenceDataset",
    "WalkForwardFold",
    "augment_windows",
    "build_sequences",
    "info_nce_loss",
    "mask_matrix",
    "masked_reconstruction_error",
    "purged_walk_forward_split",
    "r2_score",
]
