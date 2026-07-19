"""طبقة المحاكاة (Simulation Layer) — المحطة 2.

تُشتق كل المحاكيات **حصريًا** من أحداث MBO ومن إعادة بناء دفتر الأوامر، وتُنتج
ميزات كمية بترتيب زمني سببي. كل ميزة مُجمّعة على نافذة/دفعة تحمل ``availability_ts``
(زمن اكتمال النافذة) لضمان الاستخدام point-in-time دون تسريب.

المحاكيات:

* ``footprint``       — البصمة السعرية (Bid/Ask volume، Delta، Imbalance، Absorption).
* ``volume_profile``  — ملف الحجم (POC، VAH/VAL، HVN/LVN، Value Migration).
* ``order_flow``      — تدفّق الأوامر (عدوانية الشراء/البيع، OFI، استهلاك السيولة).
* ``liquidity``       — السيولة (إضافة/سحب، أوامر قائمة، كشف الآيسبرغ).
* ``auction``         — نظرية المزاد (توازن/اختلال، تمدّد، دفاع الارتداد).
* ``cross_market``    — عبر السوقين (NQ↔MNQ، Lead/Lag، تباعد، مصيدة المتداولين).
"""

from __future__ import annotations

from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades

__all__ = ["BUCKET_END", "BUCKET_START", "add_time_bucket", "extract_trades"]
