# Skenario Video: Segmentasi Paru dengan U-Net

Durasi target 2:30 sampai 2:50. Bahasa Indonesia sehari-hari. Narasi untuk ElevenLabs, visual dengan Remotion oleh model lain.

## 1. Aturan yang wajib dipatuhi

1. Semua angka harus sama dengan file ini.
2. Jangan menyebut atau menampilkan CTR 0,44 sebagai hasil pengukuran. Angka itu tidak boleh muncul di layar.
3. Dice 0,9172 selalu disebut sebagai hasil "satu pasien".
4. Tampilkan peringatan "alat bantu belajar, bukan alat diagnosis" di scene 1 dan scene 7.
5. Foto rontgen hanya dari `result_clinical_segmentation.png` atau `data/images/`. Jangan memakai foto pasien dari sumber lain.

## 2. Spesifikasi teknis

Remotion
- 1920x1080, 30 fps, keluaran mp4 (h264).
- Satu `<Sequence>` per scene. Durasi dari mp3 (`getAudioDurationInSeconds`) ditambah 0,5 detik. Timestamp di bawah perkiraan.
- Fade antar scene 10 frame. Teks layar fade dan geser naik 20 px selama 12 frame.
- Latar #0A0F1A, teks putih, aksen hijau-kebiruan (#2DD4BF) untuk mask prediksi dan kuning (#FACC15) untuk mask dokter. Judul 64 px, isi 36 px, margin 120 px.
- Teks layar adalah teks Remotion. Subtitle memakai teks narasi, maksimal dua baris.
- Salin aset ke `public/` dan panggil dengan `staticFile()`.

ElevenLabs
- Model `eleven_multilingual_v2`, suara Indonesia yang tenang. Stability 0,55, similarity 0,75, style 0.
- Satu file per scene: `s01.mp3` sampai `s07.mp3`.
- Ejaan khusus di narasi (misalnya "Yu Net", "Klahe") jangan diubah.

## 3. Aset yang boleh dipakai

| Path di folder proyek | Dipakai untuk |
| :--- | :--- |
| `result_clinical_segmentation.png` | empat panel, scene 4 dan 5 |
| `data/images/` dan `data/masks/` | contoh foto dan mask, scene 2 dan 3 |
| `radiology_ai_audit_report.json` | angka, scene 5 |

Potong `result_clinical_segmentation.png` menjadi panel-panelnya lewat CSS `object-position` dan `clip-path`, jangan dibuat ulang.

## 4. Scene

### Scene 1, 0:00 sampai 0:14, pembuka

Visual: judul "Segmentasi Paru pada Rontgen Dada". Foto rontgen dari `data/images/` muncul perlahan. Di bawah ada teks kecil peringatan.

Teks layar: Segmentasi Paru dengan U-Net. Alat bantu belajar, bukan alat diagnosis

Narasi:
> Proyek ini mengajari komputer menggambar batas paru pada foto rontgen dada. Ini alat bantu belajar, bukan alat untuk mendiagnosis.

### Scene 2, 0:14 sampai 0:35, masalah

Visual: foto rontgen dengan garis tangan bergoyang di sekitar paru, lalu tiga garis yang sedikit berbeda ditumpuk untuk menunjukkan beda antar orang.

Teks layar: Menggambar manual: lama dan berbeda antar orang

Narasi:
> Menggambar batas paru dengan tangan memakan waktu, dan hasilnya bisa beda dari satu orang ke orang lain. Kalau komputer bisa memberi batas yang konsisten, kita bisa menurunkan angka darinya, misalnya luas paru.

### Scene 3, 0:35 sampai 1:03, cara kerja

Visual: animasi alur. Foto asli, lalu foto dengan kontras lebih tajam (label "Klahe"), lalu diperkecil jadi kotak 256, lalu gambar U-Net berbentuk huruf U dengan panah sambungan skip, lalu peta peluang yang berubah jadi mask.

Teks layar: Kontras, kecilkan, U-Net, mask

Narasi:
> Pertama, kontras foto dipertajam dengan teknik yang disebut Klahe supaya batas paru lebih jelas, lalu fotonya diperkecil jadi dua ratus lima puluh enam kali dua ratus lima puluh enam. Fotonya masuk ke jaringan Yu Net. Bentuknya seperti huruf U. Sisi kiri meringkas gambar, sisi kanan membesarkannya lagi, dan ada sambungan pintas yang membawa detail dari kiri ke kanan. Keluarannya peluang paru untuk tiap piksel. Di atas lima puluh persen, piksel itu dianggap paru.

### Scene 4, 1:03 sampai 1:27, data dan pelatihan

Visual: grid 4 foto dan mask dari `data/`. Dua rumus kecil muncul sebagai ikon: "Dice" dan "BCE" bergabung jadi satu loss.

Teks layar: Montgomery County Chest X-Ray Set. 10 pasang foto dan mask di folder proyek

Narasi:
> Datanya dari Montgomery County Chest X-Ray Set. Di folder proyek ini ada sepuluh pasang foto dan mask buatan dokter. Jaringannya dilatih dengan gabungan dua loss, Dice dan BCE, yang sama-sama membantu saat paru hanya sebagian kecil dari gambar.

### Scene 5, 1:27 sampai 2:00, hasil satu pasien

Visual: `result_clinical_segmentation.png` penuh layar, lalu zoom ke panel mask dokter (kuning) dan prediksi model (hijau-kebiruan) bergantian. Angka muncul di kanan: Dice 0,9172, IoU 0,8471, 70,02 ms, 1289,1 cm persegi. Di bawahnya terlihat kotak catatan "satu pasien".

Teks layar: Satu pasien: Dice 0,9172, IoU 0,8471. 70,02 ms

Narasi:
> Pada satu pasien uji, kecocokan antara mask model dan mask dokter diukur dengan Dice dan hasilnya nol koma sembilan satu tujuh dua. Ukuran kedua, IoU, nol koma delapan empat tujuh satu. Satu foto diproses dalam sekitar tujuh puluh milidetik. Dari mask itu, luas paru dihitung sekitar seribu dua ratus delapan puluh sembilan sentimeter persegi, dengan asumsi ukuran piksel nol koma satu tujuh lima milimeter.

### Scene 6, 2:00 sampai 2:25, catatan kejujuran

Visual: teks besar bergantian. Kotak "Satu pasien saja", "Belum dicek apakah ikut dilatih", "Luas pakai asumsi ukuran piksel".

Teks layar: Satu pasien. Belum diuji pada data yang tak dipakai melatih

Narasi:
> Tapi angka ini perlu dibaca hati-hati. Itu satu pasien, dan saya belum memeriksa apakah pasien itu ikut dipakai saat melatih. Jadi ini belum membuktikan model bekerja pada pasien baru. Luas paru juga memakai asumsi ukuran piksel, bukan ukuran asli dari berkas gambar.

### Scene 7, 2:25 sampai 2:45, langkah berikutnya dan penutup

Visual: daftar dua poin, dan peringatan medis di bawah.

Teks layar: Berikutnya: data uji terpisah, lebih dari satu foto. Alat bantu belajar, bukan alat diagnosis

Narasi:
> Langkah berikutnya, menguji dengan beberapa foto yang tidak dipakai untuk melatih. Sekali lagi, ini proyek belajar dan bukan alat diagnosis. Terima kasih sudah menonton.

## 5. Daftar periksa sebelum render

1. Angka cocok: 256x256, 0,5, 10 pasang, 0,9172, 0,8471, 70,02 ms, 1289,1 cm persegi, 0,175 mm.
2. Tidak ada tulisan CTR, kardiomegali, atau "normal" di layar atau narasi.
3. Peringatan medis tampil di scene 1 dan 7. Catatan "satu pasien" tampil selama scene 5 dan 6.
4. Subtitle sama dengan narasi. Total 2:30 sampai 2:50.
