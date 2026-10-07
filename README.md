# EMA Lightning sesli asistan

Bu makinede çalışan Türkçe sesli asistan denemesi. Dinleme (Zipformer), konuşma (EMA Lightning), ses etkinliği (Silero VAD) ve tur sonu algılama (LiveKit `v1-mini`) yereldir. Cevabı şimdilik Gemini 3.1 Flash-Lite yazar. Ses tarayıcıya WebRTC ile, kendi makinemizde çalışan LiveKit sunucusu üzerinden gider.

## Kurulum

Python 3.12 gerekir. GTX 1070 gibi eski NVIDIA kartlarda PyTorch CUDA 11.8 tekerleği kullanılır:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend; npm install; cd ..
```

LiveKit sunucusunu [GitHub sürümlerinden](https://github.com/livekit/livekit/releases) indirip `bin\livekit-server.exe` olarak koyun (Windows için `windows_amd64.zip`).

`.env.example` dosyasını `.env` olarak kopyalayıp `GEMINI_API_KEY` değerini yazın.

## Çalıştırma

Dört ayrı terminal:

```powershell
.\bin\livekit-server.exe --dev --bind 127.0.0.1 --node-ip 127.0.0.1
.\.venv\Scripts\python.exe agent\worker.py start
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
cd frontend; npm run dev
```

Arayüz `http://127.0.0.1:5173` adresindedir. **Canlı** sekmesi LiveKit görüşmesidir; **Tur tur** sekmesi eski bas-konuş denemesidir.

## Araya girme

EMA konuşurken gelen söz `agent/bargein.py` içindeki kurallarla değerlendirilir:

- "hıhı", "tamam", "evet" gibi onaylar EMA'yı susturmaz ve cevap üretmez.
- EMA soru sorduysa "evet", "hayır" gibi cevaplar EMA'yı susturur ve cevaba geçilir.
- "dur", "bekle", "bir saniye" hemen susturur.
- Hakem, `.env` içinde `OPENROUTER_API_KEY` varsa OpenRouter üzerindeki TypeSafe Jev'dir (`typesafe/jev-1.13`, yaklaşık 300 ms); yoksa Gemini'dir. Jev metin üretmez, "müşteri sözü kesmek istiyor mu?" sorusuna evet olasılığı döner; %50 ve üstü keser.
- Bunların dışındaki her söz, ses yazıya döküldükçe her yeni parçasıyla hakeme sorulur. Aynı anda en fazla iki soru açık kalır, fazlası en yeni metinle sıraya girer. Ana akış hakemi beklemez.
- Hakem 1,5 saniyede cevap vermezse kural karar verir: üç kelime ya da 1,2 saniyeden uzun konuşma susturur, EMA'nın kendi cümlesinin yankısı ve kısa tek kelimelik sesler yok sayılır.
- `.env` içine `BARGEIN_MODE=rules` yazılırsa kurala uyan sözler hakeme sorulmadan anında karara bağlanır; hakem yalnızca belirsiz sözlere bakar.
- 30 saniyede üç kesmeden sonra, ya da 6 saniyeden uzun üstüne konuşmada EMA artık susmaz. Cümlesini bitirip kibarca uyarır. Uyarıdan sonra 60 saniye tekrar uyarmaz.

Kural testleri:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```
