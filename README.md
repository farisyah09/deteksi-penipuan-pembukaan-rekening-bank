# Deteksi Penipuan Pembukaan Rekening (BAF Base, month 0-3)

Notebook: `kasus_A_fraud_baf.ipynb` (jalankan *Run All*, atas ke bawah). Hanya butuh scikit-learn.

**Setup:** `pip install -r requirements.txt`, taruh `Base.csv` (Kaggle BAF, varian Base) di `data/Base.csv`.
**Output:** `models/fraud_model.joblib` dan `models/fraud_model_meta.json` (fitur, threshold, skor). Muat ulang dengan `joblib.load`; notebook memverifikasi hasilnya sama dan menyediakan fungsi `score_applications`.
Uji cepat tanpa data: `SMOKE_TEST=1` (data sintetis, skor tidak bermakna).

## Data
Hanya baris `month` 0-3 (~1% penipu). Sesuai datasheet BAF, nilai -1 (negatif pada `intended_balcon_amount`) berarti data hilang, diubah ke NaN. `device_fraud_count` (konstan 0) dibuang.

## Metode dan alasan
HistGradientBoostingClassifier (kategorik dan NaN native, early stopping). Data tabular campuran dengan banyak nilai hilang dan kelas sangat tidak seimbang; boosting pohon menangani ini tanpa scaling. Tanpa resampling/bobot kelas karena metrik berbasis peringkat skor, dan threshold ditentukan langsung dari FPR 5%.

## Pembagian data
Temporal: **latih = month 0-2, validasi = month 3.** Model dipakai untuk pengajuan masa depan dan pola penipuan bergeser antar bulan; split acak mencampur bulan yang sama sehingga skor terlalu optimis. Satu baris = satu pengajuan, tidak ada baris ganda lintas split.

## Skor di data validasi (month 3)
- **TPR pada FPR 5%: `___`**
- ROC-AUC: `___`

(isi dari output notebook; threshold FPR 5% tersimpan di meta)
