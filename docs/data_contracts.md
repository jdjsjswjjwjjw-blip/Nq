# عقود البيانات (Data Contracts)

المصدر الوحيد للحقيقة في النظام هو تدفّق **MBO (Market By Order)**. يحدّد هذا
المستند العقد البنيوي الملزِم لكل بيانات تدخل النظام، ويُفرَض برمجيًا عبر
`nq.contracts`.

## الحقول الزمنية القانونية (Canonical Temporal Fields)

| الحقل | النوع | الوصف | القاعدة الحاكمة |
|-------|------|-------|------------------|
| `event_ts` | `Int64` (ns) | زمن وقوع الحدث في السوق | أساس الترتيب السببي |
| `ingest_ts` | `Int64` (ns) | زمن استلام الحدث لدينا | `ingest_ts >= event_ts` |
| `sequence` | `UInt64` | تسلسل رتيب لفضّ التعادل | يكمّل الترتيب السببي |
| `availability_ts` | `Int64` (ns) | زمن إتاحة الميزة/المخرَج | `availability_ts >= event_ts` |

الترتيب السببي القانوني هو `(event_ts, sequence)`. أي حساب لاحق يرى الأحداث
بهذا الترتيب حصريًا.

## مخطط MBO (`MBO_SCHEMA`)

| العمود | النوع | الوصف |
|--------|------|-------|
| `event_ts` | `Int64` | زمن الحدث (ns) |
| `ingest_ts` | `Int64` | زمن الاستلام (ns) |
| `sequence` | `UInt64` | التسلسل الرتيب |
| `instrument_id` | `UInt32` | معرّف الأداة |
| `symbol` | `Utf8` | الرمز (NQ / MNQ) |
| `action` | `Enum` | نوع الحدث (`A/C/M/R/T/F/N`) |
| `side` | `Enum` | الجانب (`B/A/N`) |
| `price` | `Int64` | سعر بنقطة ثابتة (× `PRICE_SCALE = 1e-9`) |
| `size` | `UInt32` | الحجم |
| `order_id` | `UInt64` | معرّف الأمر |
| `flags` | `UInt8` | أعلام السوق (bit flags) |

### أنواع الأحداث (`MboAction`)

| الرمز | المعنى |
|------|--------|
| `A` | إضافة أمر (Add) |
| `C` | إلغاء (Cancel) |
| `M` | تعديل (Modify) |
| `R` | مسح الدفتر (Clear/Reset) |
| `T` | صفقة (Trade) |
| `F` | تنفيذ (Fill) |
| `N` | لا فعل (None/heartbeat) |

### الجوانب (`MboSide`)

| الرمز | المعنى |
|------|--------|
| `B` | طلب (Bid) |
| `A` | عرض (Ask) |
| `N` | غير محدد (None) |

## التحقق (Validation)

`validate_mbo_frame(frame)` يفرض: اكتمال الأعمدة، تطابق الأنواع، وسلامة النقطة
الزمنية (`ingest_ts >= event_ts`). يرفع `ValueError` عند أي خرق للعقد.

## الأسعار كنقطة ثابتة (Fixed-Point Prices)

تُخزّن الأسعار كأعداد صحيحة لتفادي أخطاء الفاصلة العائمة وضمان الدقّة الكمية:

```text
real_price = price * PRICE_SCALE   # PRICE_SCALE = 1e-9
```
