"""مخزن الميزات point-in-time (Feature Store) — المحطة 3.

يوحّد مخرجات كل المحاكيات في مخطط زمني قانوني واحد، ويتيح:

* استرجاعًا زمنيًا دقيقًا (time-travel): قيمة كل ميزة كما كانت متاحة في زمن ``t``.
* دمج point-in-time (as-of backward) لبناء مصفوفات تدريب بلا تسريب مستقبلي.
* إصدارات (versioning) للميزات، وحفظًا/قراءة عموديًا (Parquet).

المبدأ الحاكم: لا تُسترجَع أي قيمة إلا إذا كان ``availability_ts <= t`` (نقطة زمنية).
"""

from __future__ import annotations

from nq.features.store import (
    FEATURE_STORE_SCHEMA,
    FeatureStore,
    wide_to_features,
)

__all__ = ["FEATURE_STORE_SCHEMA", "FeatureStore", "wide_to_features"]
