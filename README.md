# EMA Lightning model test

Türkçe metni bu makinede sese çeviren küçük bir deneme. Arayüz React, servis FastAPI. Model `ema-lightning` paketidir; ses bilgisayardan çıkmaz.

## Çalıştırma

Python 3.12 gerekir. GTX 1070 gibi eski NVIDIA kartlarda PyTorch CUDA 11.8 tekerleği kullanılır:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

Ayrı bir terminalde:

```powershell
cd frontend
npm install
npm run dev
```

Arayüz `http://127.0.0.1:5173` adresindedir. API `8010` portundadır.

Modelde tek ses vardır. Hız, örnekleme ve seed arayüzden değişir.
