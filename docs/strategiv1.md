# Strategi Klasifikasi 3 Kelas Berbasis PHQ-8

full pake dataset DAIC-WOZ aja

### Perbaikan Aturan Logika Hierarkis (Rule-Based) yang Memanfaatkan Kolom Individual

Agar pelabelan akurat secara data sains dan mencegah *overlap* (tumpang tindih label) serta mencegah adanya data *null* (tanpa label), aturan ini menggunakan pendekatan **Hierarki / Prioritas (If-Elif-Else)**. Setiap baris data akan dievaluasi secara berurutan dan hanya akan mendapat **satu label yang paling dominan**.

Berikut adalah hierarki logika klasifikasi:

**1. Prioritas Pertama: DEPRESI (Kondisi Klinis Terberat)**
- `PHQ8_Score` $\ge$ 10 (standar klinis depresi *moderate-severe*) OR
- Nilai `PHQ8_Depressed` $\ge$ 2 AND `PHQ8_NoInterest` $\ge$ 2 AND `PHQ8_Failure` $\ge$ 2.
- *Logika: Kondisi paling parah dievaluasi pertama. Jika lolos evaluasi ini, data otomatis dilabeli DEPRESI dan evaluasi berhenti.*

**2. Prioritas Kedua: CEMAS (Gejala Agitasi dan Gangguan Fokus)**
- `PHQ8_Score` $\ge$ 5 AND
- Nilai `PHQ8_Moving` $\ge$ 1 (mengalami agitasi/gelisah) ATAU `PHQ8_Concentrating` $\ge$ 2.
- *Logika: Data yang masuk ke tahap ini skornya sudah pasti di bawah 10 (bukan depresi mayor). Jika ada tanda agitasi pikiran/fisik, diklasifikasikan sebagai CEMAS.*  

**3. Prioritas Ketiga: STRES (Gejala Keluhan Fisik dan Kelelahan)**
- `PHQ8_Score` $\ge$ 5 AND
- Nilai `PHQ8_Sleep` $\ge$ 2 ATAU `PHQ8_Tired` $\ge$ 2.
- *Logika: Tidak ada gejala kecemasan, tapi tubuh mengalami kelelahan dan gangguan tidur. Ini merupakan indikasi dominan STRES fisik.*

**4. Prioritas Keempat: NORMAL**
- `PHQ8_Score` $\le$ 4 AND
- Kolom inti depresi (`PHQ8_Depressed`, `PHQ8_NoInterest`) harus bernilai maksimal 1.
- *Logika: Skor sangat rendah dan gejala afektif minim, menandakan kondisi psikologis yang stabil/sehat.*

**5. Kondisi Fallback / Catch-All (Menangani Data Sisa)**
- *Kriteria*: Jika sebuah baris data gagal memenuhi keempat kriteria di atas (misal: skor 5-9 tapi gejalanya tersebar tipis, misal tidur=1, lelah=1, murung=1, dll).
- *Tindakan*: Secara otomatis dikelompokkan ke dalam kategori **STRES** (sebagai penanda adanya beban psikologis ringan).
- *Logika: Memastikan 100% data terisi (tidak ada nilai NULL) sehingga dataset aman dan valid saat di-training oleh algoritma Machine Learning.*