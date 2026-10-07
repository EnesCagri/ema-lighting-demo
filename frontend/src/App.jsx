import { useEffect, useRef, useState } from 'react'

const EXAMPLES = [
  { label: 'Karşılama', text: 'Migros Sanal Market\'e hoş geldiniz. Size nasıl yardımcı olabilirim?' },
  { label: 'Sipariş', text: 'Siparişiniz hazırlanıyor. Teslimatınız bugün saat 14:00 ile 16:00 arasındadır.' },
  { label: 'Kurye', text: 'Kuryeniz yola çıktı. Tahmini varış süreniz 25 dakikadır.' },
  { label: 'Tutar', text: 'Kapıda ödeme tutarınız 847 lira 50 kuruştur. Nakit ya da kredi kartı ile ödeyebilirsiniz.' },
  { label: 'Stok', text: 'Bir ürün stokta olmadığı için siparişinizden çıkarıldı. Fark tutarı kartınıza iade edilecektir.' },
  { label: 'İptal', text: 'İptal işleminiz tamamlandı. Ücret 3 iş günü içinde kartınıza yansır.' },
  { label: 'Adres', text: 'Teslimat için kapı kodunuzu ve daire numaranızı söyler misiniz?' },
  { label: 'Aktarma', text: 'Sizi ilgili birime aktarıyorum. Lütfen hatta kalın.' },
]

const RATES = [
  { value: 48000, label: '48 kHz' },
  { value: 24000, label: '24 kHz' },
  { value: 16000, label: '16 kHz' },
  { value: 8000, label: '8 kHz' },
]

export default function App() {
  const [text, setText] = useState(EXAMPLES[0].text)
  const [speed, setSpeed] = useState(1)
  const [sampleRate, setSampleRate] = useState(48000)
  const [seed, setSeed] = useState(0)
  const [ready, setReady] = useState(false)
  const [device, setDevice] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [audioUrl, setAudioUrl] = useState('')
  const [info, setInfo] = useState(null)
  const audioRef = useRef(null)
  const urlRef = useRef('')

  useEffect(() => {
    let stop = false
    async function poll() {
      try {
        const response = await fetch('/api/health')
        if (!response.ok) return
        const data = await response.json()
        if (stop) return
        setReady(data.ready)
        setDevice(data.device || '')
      } catch {
        if (!stop) setReady(false)
      }
    }
    poll()
    const timer = setInterval(poll, 1000)
    return () => {
      stop = true
      clearInterval(timer)
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    }
  }, [])

  async function speak(nextText = text) {
    const spoken = nextText.trim()
    if (!spoken || busy) return
    setBusy(true)
    setError('')
    try {
      const response = await fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: spoken,
          speed: Number(speed),
          seed: Number(seed),
          sample_rate: sampleRate,
        }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Ses üretilemedi.')
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      urlRef.current = url
      setAudioUrl(url)
      setInfo({
        duration: response.headers.get('X-Duration'),
        generateMs: response.headers.get('X-Generate-Ms'),
        rate: response.headers.get('X-Sample-Rate'),
        seed: response.headers.get('X-Seed'),
      })
      requestAnimationFrame(() => {
        audioRef.current?.play().catch(() => {})
      })
    } catch (err) {
      setError(err.message || 'Ses üretilemedi.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="app">
      <header>
        <h1>EMA <span>Lightning</span> model test</h1>
        <div className="status">
          {ready ? `Hazır · ${device}` : 'Model yükleniyor…'}
        </div>
      </header>

      <div className="layout">
        <section className="panel">
          <label className="block" htmlFor="metin">Metin</label>
          <textarea
            id="metin"
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) speak()
            }}
            placeholder="Çağrı metnini yazın"
          />
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button
                key={example.label}
                type="button"
                className="chip"
                onClick={() => {
                  setText(example.text)
                  speak(example.text)
                }}
              >
                {example.label}
              </button>
            ))}
          </div>
          <button className="speak" type="button" disabled={!ready || busy || !text.trim()} onClick={() => speak()}>
            {busy ? 'Üretiliyor…' : 'Sese çevir'}
          </button>
          <p className="error">{error}</p>
          {audioUrl && (
            <div className="player">
              <audio ref={audioRef} src={audioUrl} controls autoPlay />
              {info && (
                <div className="meta">
                  {Number(info.duration).toFixed(2)} sn ses · {Number(info.generateMs).toFixed(0)} ms üretim · {info.rate} Hz · seed {info.seed}
                </div>
              )}
              <a href={audioUrl} download="ema.wav">WAV indir</a>
            </div>
          )}
        </section>

        <aside className="panel">
          <label className="block">Ses</label>
          <div className="voice">
            <strong>EMA</strong>
          </div>

          <div className="row">
            <label className="block" htmlFor="hiz">Hız</label>
            <b>{Number(speed).toFixed(2)}×</b>
          </div>
          <input
            id="hiz"
            type="range"
            min="0.25"
            max="4"
            step="0.05"
            value={speed}
            onChange={(event) => setSpeed(event.target.value)}
          />
          <div className="choices">
            {[0.75, 1, 1.25, 1.5].map((value) => (
              <button
                key={value}
                type="button"
                className={Number(speed) === value ? 'choice on' : 'choice'}
                onClick={() => setSpeed(value)}
              >
                {value}×
              </button>
            ))}
          </div>

          <label className="block">Örnekleme</label>
          <div className="choices">
            {RATES.map((rate) => (
              <button
                key={rate.value}
                type="button"
                className={sampleRate === rate.value ? 'choice on' : 'choice'}
                onClick={() => setSampleRate(rate.value)}
              >
                {rate.label}
              </button>
            ))}
          </div>

          <label className="block" htmlFor="seed">Seed</label>
          <div className="seed">
            <input
              id="seed"
              type="number"
              min="0"
              step="1"
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
            />
            <button
              type="button"
              className="ghost"
              onClick={() => setSeed(Math.floor(Math.random() * 1_000_000))}
            >
              Rastgele
            </button>
          </div>
        </aside>
      </div>
    </main>
  )
}
