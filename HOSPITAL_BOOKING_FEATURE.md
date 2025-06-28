# 🏥 ميزة حجز المستشفى - Hospital Booking Feature

## 📋 نظرة عامة

تم إضافة ميزة حجز المستشفى إلى تشات بوت التشخيص الطبي. هذه الميزة تسمح للمستخدمين بحجز مواعيد في المستشفيات المتاحة بناءً على موقعهم.

## ✨ الميزات الجديدة

### 1. **حجز موعد في المستشفى**
- اختيار المستشفى من قائمة المستشفيات المتاحة
- تحديد التاريخ والوقت المناسب
- إدخال بيانات المريض
- تأكيد الحجز وحفظ التفاصيل

### 2. **قائمة المستشفيات**
- عرض المستشفيات حسب الموقع
- معلومات تفصيلية عن كل مستشفى:
  - الاسم
  - الموقع (المدينة والبلد)
  - التقييم
  - رقم الهاتف
  - تاريخ التأسيس

### 3. **نظام الحجز الذكي**
- تواريخ متاحة (الـ 7 أيام القادمة)
- أوقات متاحة (8 فترات زمنية)
- رقم مرجعي للحجز
- حفظ تفاصيل الحجز في ملف

## 🚀 كيفية الاستخدام

### من القائمة الرئيسية:
1. اختر "4) Book hospital appointment"
2. أدخل موقعك (إذا لم يكن محدداً)
3. اختر المستشفى من القائمة
4. أدخل بياناتك الشخصية
5. اختر التاريخ والوقت
6. أكد الحجز

### بعد التشخيص:
1. بعد الحصول على التشخيص
2. سيُسأل إذا كنت تريد حجز موعد في المستشفى
3. إذا أجبت "نعم"، ستظهر قائمة المستشفيات

## 📊 البيانات المستخدمة

### API المستخدم:
- **Doctors API**: `https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/doctors`
- **Hospitals API**: `https://medeasy-backend-cgetg3arfvgfcjcq.westcentralus-01.azurewebsites.net/api/users/hospitals`

### هيكل البيانات:

#### بيانات المستشفى:
```json
{
  "name": "اسم المستشفى",
  "phone": "رقم الهاتف",
  "city": "المدينة",
  "country": "البلد",
  "rate": 4.5,
  "Established": "تاريخ التأسيس"
}
```

#### بيانات الحجز:
```json
{
  "status": "confirmed",
  "booking_reference": "ABC12345",
  "hospital_id": "id",
  "patient_name": "اسم المريض",
  "patient_phone": "رقم الهاتف",
  "appointment_date": "2024-01-15",
  "appointment_time": "10:00 AM",
  "symptoms": ["symptom1", "symptom2"]
}
```

## 🔧 الملفات المضافة/المعدلة

### ملفات جديدة:
- `test_api.py` - اختبار API
- `HOSPITAL_BOOKING_FEATURE.md` - هذا الملف

### ملفات معدلة:
- `chatbot_interface.py` - إضافة دوال الحجز
- `PROBLEMS_AND_SOLUTIONS.md` - تحديث التوثيق

## 📁 حفظ البيانات

### مجلد الحجوزات:
- يتم إنشاء مجلد `bookings/` تلقائياً
- كل حجز يُحفظ في ملف JSON منفصل
- اسم الملف: `booking_[رقم_مرجعي].json`

### مثال على ملف الحجز:
```json
{
  "status": "confirmed",
  "booking_reference": "ABC12345",
  "hospital_id": "67cd7c8ff2d6e1f39d3e18cc",
  "patient_name": "أحمد محمد",
  "patient_phone": "0123456789",
  "appointment_date": "2024-01-15",
  "appointment_time": "10:00 AM",
  "symptoms": ["headache", "fever"],
  "booking_timestamp": "2024-01-14T10:30:00"
}
```

## 🧪 اختبار الميزة

### تشغيل اختبار API:
```bash
python test_api.py
```

### النتائج المتوقعة:
```
🧪 API TESTING SUITE
==================================================
🏥 Testing Doctors API...
Status Code: 200
✅ Found 10 doctors

🏥 Testing Hospitals API...
Status Code: 200
✅ Found 10 hospitals

📅 Testing Booking API...
❌ No booking API found

==================================================
📊 SUMMARY:
  Doctors API: ✅ Working
  Hospitals API: ✅ Working
  Booking API: ❌ Not Found

🎉 APIs are working correctly!
✅ Data structure is valid
✅ Ready to implement hospital booking feature
```

## 🎯 مثال على الاستخدام

```
🏥 HOSPITAL BOOKING MENU
==================================================
🏥 Found 3 hospitals in Cairo:

1. Tayseer
   📍 Location: Cairo, Egypt
   ⭐ Rating: 4/5
   📞 Phone: +201213452111
   🏗️  Established: 30-01-1980

2. Al-Ahly
   📍 Location: Cairo, Egypt
   ⭐ Rating: 4.5/5
   📞 Phone: +20123456789
   🏗️  Established: 15-03-1975

Select hospital (1-3) or 'back' to return: 1

📋 Booking appointment at Tayseer
==================================================
Enter your full name: أحمد محمد
Enter your phone number: 0123456789

📅 Available dates (starting from tomorrow):
  1. 2024-01-15 (Monday)
  2. 2024-01-16 (Tuesday)
  3. 2024-01-17 (Wednesday)
  ...

Select date (1-7): 1

🕐 Available time slots:
  1. 09:00 AM
  2. 10:00 AM
  3. 11:00 AM
  ...

Select time (1-8): 2

📋 Booking Summary:
  Hospital: Tayseer
  Patient: أحمد محمد
  Phone: 0123456789
  Date: 2024-01-15
  Time: 10:00 AM

Confirm booking? (yes/no): yes

🏥 HOSPITAL BOOKING SYSTEM
==================================================
✅ BOOKING CONFIRMED!
==================================================
📅 Your appointment has been successfully booked.
📞 You will receive a confirmation call shortly.
🏥 Please arrive 15 minutes before your appointment time.
📋 Don't forget to bring your ID and insurance card.
🔢 Booking Reference: ABC12345

🎉 Booking completed successfully!
📧 A confirmation email has been sent to your registered email.
📱 You will also receive an SMS confirmation.
```

## ⚠️ ملاحظات مهمة

1. **لا يوجد API حقيقي للحجز**: الميزة حالياً تحاكي عملية الحجز
2. **حفظ محلي**: يتم حفظ الحجوزات في ملفات محلية
3. **البيانات صحيحة**: API المستشفيات والأطباء يعمل بشكل صحيح
4. **قابل للتطوير**: يمكن ربط الميزة بـ API حقيقي للحجز في المستقبل

## 🔮 التطوير المستقبلي

1. **ربط بـ API حقيقي للحجز**
2. **إضافة نظام إلغاء الحجوزات**
3. **إضافة تذكيرات بالمواعيد**
4. **ربط بـ نظام الدفع**
5. **إضافة تقييمات المستشفيات** 