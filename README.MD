# 🧴 Skinthesia Recommendation API

API berbasis Flask yang memanfaatkan model machine learning untuk merekomendasikan produk skincare berdasarkan jenis kulit, masalah kulit, tujuan perawatan, dan preferensi pengguna.

## 🚀 Fitur

* 🔍 Prediksi kategori bahan aktif berdasarkan input pengguna menggunakan model klasifikasi multilabel TensorFlow.
* 🧠 Mapping otomatis dari kategori bahan ke bahan spesifik (contoh: `brightening` → `niacinamide`, `vitamin c`, dll).
* 🧴 Rekomendasi produk skincare berbasis:

  * Umur pengguna
  * Rentang harga
  * Kategori produk
  * Kecocokan bahan aktif
* 📊 Filtering berdasarkan review score dan total ulasan.

## 🛠️ Instalasi

### 1. Clone repositori

```bash
git clone https://github.com/jasmeinalbr/skinthesia-api
cd skinthesia-api
```

### 2. Install dependency

```bash
pip install -r requirements.txt
```

> **Catatan:** Pastikan Anda menggunakan Python 3.8+ dan sudah mengaktifkan virtual environment.

### 3. Struktur direktori

```
skincare-recommender/
│
├── src/
│   └── models/
│       ├── ingredients_category_classification_model.keras
│       └── mlb_classes.json
│
├── data/
│   └── products_integrated_features.csv
│
├── logs/
│   └── recommendation_model.log
│
└── script.py 
```

## 🧪 Menjalankan Aplikasi

```bash
python script.py
```

Secara default, Flask akan berjalan di `http://127.0.0.1:5000`.

## 📮 Endpoint

### `POST /recommend`

Menerima data pengguna dan memberikan rekomendasi produk skincare.

#### 🔸 Request JSON Body

```json
{
  "skin_type": ["oily"],
  "skin_concern": ["acne"],
  "skin_goal": ["brightening"],
  "ingredient": ["niacinamide"],
  "age": "19-24",
  "price_min": 50000,
  "price_max": 200000,
  "category": "Serum",
  "best_thresholds": [0.5, 0.5, 0.5, 0.45, 0.5, 0.45]
}
```

#### 🔸 Response (200 OK)

```json
{
  "status": "success",
  "products": [
    {
      "product_name": "Brightening Serum X",
      "brand": "BrandA",
      "image": "https://example.com/image.jpg",
      "price": 175000,
      "rating": 4.5,
      "total_reviews": 215,
      "ingredients": "niacinamide, vitamin c"
    },
    ...
  ]
}
```

#### 🔸 Response Jika Tidak Ada Produk

```json
{
  "status": "success",
  "products": [],
  "message": "Tidak ada produk yang cocok dengan kriteria. Kemungkinan penyebab: ..."
}
```

## 🧠 Tentang Model

Model klasifikasi dikembangkan menggunakan TensorFlow untuk memprediksi kategori bahan (`ingredient_category`) berdasarkan:

* Tipe kulit
* Masalah kulit
* Tujuan skincare
* Bahan yang sudah diketahui

Model menggunakan loss function kustom `weighted_binary_crossentropy` untuk menangani data imbalance.

## 📝 Logging

Semua proses akan tercatat dalam file `logs/recommendation_model.log` dengan detail seperti:

* Input normalisasi
* Probabilitas prediksi
* Filter yang diterapkan
* Jumlah produk hasil filter
* Error detail jika terjadi

## 📌 Catatan Penting

* Dataset CSV harus memiliki kolom: `product_name`, `brand`, `image`, `price`, `rating`, `total_reviews`, `age`, `category`, `ingredients`.
* File model dan `mlb_classes.json` harus sesuai struktur dan berada di direktori `src/models/`.

## 🤝 Kontribusi

Pull request dan diskusi sangat dipersilakan! Silakan fork dan modifikasi sesuai kebutuhan.

