import { useEffect, useRef, useState } from 'react'
import Live from './Live.jsx'

const RATES = [
  { value: 48000, label: '48 kHz' },
  { value: 24000, label: '24 kHz' },
  { value: 16000, label: '16 kHz' },
  { value: 8000, label: '8 kHz' },
]

export default function App() {
  const [tab, setTab] = useState('live')
  return (
    <main className="app">
      <header>
        <h1>EMA <span>Lightning</span> sesli asistan</h1>
        <nav className="tabs">
          <button type="button" className={tab === 'live' ? 'choice on' : 'choice'} onClick={() => setTab('live')}>
            Canlı
          </button>
          <button type="button" className={tab === 'turns' ? 'choice on' : 'choice'} onClick={() => setTab('turns')}>
            Tur tur
          </button>
        </nav>
      </header>
      {tab === 'live' ? <Live /> : <TurnTest />}
    </main>
  )
}

function TurnTest() {
  const [draft, setDraft] = useState('')
  const [speed, setSpeed] = useState(1)
  const [sampleRate, setSampleRate] = useState(48000)
  const [seed, setSeed] = useState(0)
  const [ready, setReady] = useState(false)
  const [sttReady, setSttReady] = useState(false)
  const [agentReady, setAgentReady] = useState(false)
  const [model, setModel] = useState('gemini-3.1-flash-lite')
  const [device, setDevice] = useState('')
  const [turns, setTurns] = useState([])
  const [busy, setBusy] = useState(false)
  const [recording, setRecording] = useState(false)
  const [hearing, setHearing] = useState(false)
  const [follow, setFollow] = useState(true)
  const [error, setError] = useState('')
  const [audioUrl, setAudioUrl] = useState('')
  const [info, setInfo] = useState(null)
  const audioRef = useRef(null)
  const urlRef = useRef('')
  const captureRef = useRef(null)
  const turnsRef = useRef([])
  const followRef = useRef(true)
  const busyRef = useRef(false)
  const armedRef = useRef(false)
  const listRef = useRef(null)

  useEffect(() => {
    let stop = false
    async function poll() {
      try {
        const response = await fetch('/api/health')
        if (!response.ok) return
        const data = await response.json()
        if (stop) return
        setReady(data.ready)
        setSttReady(Boolean(data.stt))
        setAgentReady(Boolean(data.agent))
        setModel(data.model || 'gemini-3.1-flash-lite')
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

  useEffect(() => {
    followRef.current = follow
  }, [follow])

  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [turns, busy, hearing])

  function playWav(base64) {
    const binary = atob(base64)
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
    const url = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }))
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    urlRef.current = url
    setAudioUrl(url)
    armedRef.current = true
    requestAnimationFrame(() => {
      audioRef.current?.play().catch(() => {})
    })
  }

  async function askAgent(userText, sttMs = null) {
    const spoken = userText.trim()
    if (!spoken) throw new Error('Konuşma algılanmadı.')
    const history = turnsRef.current.slice(-8)
    turnsRef.current = [...turnsRef.current, { role: 'user', text: spoken }]
    setTurns(turnsRef.current)
    setBusy(true)
    setError('')
    try {
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: spoken,
          history,
          speed: Number(speed),
          seed: Number(seed),
          sample_rate: sampleRate,
        }),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.detail || 'Cevap üretilemedi.')
      turnsRef.current = [...turnsRef.current, { role: 'assistant', text: body.reply }]
      setTurns(turnsRef.current)
      setInfo({
        sttMs,
        llmMs: body.llm_ms,
        generateMs: body.generate_ms,
        duration: body.duration,
        model: body.model,
      })
      playWav(body.audio)
    } catch (err) {
      turnsRef.current = turnsRef.current.slice(0, -1)
      setTurns(turnsRef.current)
      setError(err.message || 'Cevap üretilemedi.')
      throw err
    } finally {
      setBusy(false)
    }
  }

  function encodeWav(samples, rate) {
    const bytes = new ArrayBuffer(44 + samples.length * 2)
    const view = new DataView(bytes)
    const write = (offset, value) => {
      for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i))
    }
    write(0, 'RIFF')
    view.setUint32(4, 36 + samples.length * 2, true)
    write(8, 'WAVE')
    write(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, 1, true)
    view.setUint32(24, rate, true)
    view.setUint32(28, rate * 2, true)
    view.setUint16(32, 2, true)
    view.setUint16(34, 16, true)
    write(36, 'data')
    view.setUint32(40, samples.length * 2, true)
    let offset = 44
    for (let i = 0; i < samples.length; i += 1, offset += 2) {
      const sample = Math.max(-1, Math.min(1, samples[i]))
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
    }
    return new Blob([bytes], { type: 'audio/wav' })
  }

  async function startListening() {
    if (captureRef.current || busyRef.current) return
    setError('')
    let mic
    try {
      mic = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('Mikrofona izin verilemedi.')
      followRef.current = false
      setFollow(false)
      return
    }
    const context = new AudioContext()
    const source = context.createMediaStreamSource(mic)
    const processor = context.createScriptProcessor(4096, 1, 1)
    const silent = context.createGain()
    silent.gain.value = 0
    const capture = {
      mic,
      context,
      processor,
      chunks: [],
      sampleRate: context.sampleRate,
      closed: false,
      heard: false,
      silence: 0,
      seconds: 0,
    }
    processor.onaudioprocess = (event) => {
      if (capture.closed) return
      const data = event.inputBuffer.getChannelData(0)
      capture.chunks.push(new Float32Array(data))
      capture.seconds += data.length / capture.sampleRate
      if (!followRef.current) {
        if (capture.seconds > 20) stopAndReply()
        return
      }
      let sum = 0
      for (let i = 0; i < data.length; i += 1) sum += data[i] * data[i]
      const rms = Math.sqrt(sum / data.length)
      if (rms > 0.02) {
        capture.heard = true
        capture.silence = 0
      } else if (capture.heard) {
        capture.silence += data.length / capture.sampleRate
      }
      if ((capture.heard && capture.silence > 0.75) || capture.seconds > 20) stopAndReply()
    }
    source.connect(processor)
    processor.connect(silent)
    silent.connect(context.destination)
    captureRef.current = capture
    setRecording(true)
  }

  async function stopAndReply() {
    const capture = captureRef.current
    if (!capture || capture.closed) return
    capture.closed = true
    captureRef.current = null
    setRecording(false)
    capture.processor.disconnect()
    capture.mic.getTracks().forEach((track) => track.stop())
    const length = capture.chunks.reduce((sum, chunk) => sum + chunk.length, 0)
    const samples = new Float32Array(length)
    let offset = 0
    capture.chunks.forEach((chunk) => {
      samples.set(chunk, offset)
      offset += chunk.length
    })
    await capture.context.close()
    setHearing(true)
    setError('')
    try {
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: encodeWav(samples, capture.sampleRate),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.detail || 'Konuşma yazıya çevrilemedi.')
      if (!body.text) {
        setHearing(false)
        if (followRef.current) {
          setError('Duyulmadı, tekrar dinleniyor.')
          startListening()
          return
        }
        throw new Error('Konuşma algılanmadı.')
      }
      setHearing(false)
      await askAgent(body.text, body.stt_ms)
    } catch (err) {
      if (err.message) setError(err.message)
      setHearing(false)
      setBusy(false)
    }
  }

  async function sendTyped(event) {
    event.preventDefault()
    const spoken = draft.trim()
    if (!spoken || busy) return
    setDraft('')
    try {
      await askAgent(spoken)
    } catch {
      setDraft(spoken)
    }
  }

  function resetChat() {
    turnsRef.current = []
    setTurns([])
    setError('')
    setInfo(null)
    armedRef.current = false
  }

  function onPlaybackEnded() {
    if (!armedRef.current || !followRef.current || busyRef.current || captureRef.current) return
    window.setTimeout(() => {
      if (armedRef.current && followRef.current && !busyRef.current && !captureRef.current) startListening()
    }, 300)
  }

  const status = !ready || !sttReady
    ? 'Modeller yükleniyor…'
    : agentReady
      ? `Hazır · ${device} · ${model}`
      : `Hazır · ${device} · Gemini anahtarı bekleniyor`

  return (
    <>
      <div className="status">{status}</div>
      <div className="layout">
        <section className="panel">
          <div className="transcript" ref={listRef}>
            {turns.length === 0 && (
              <p className="empty">Konuşun veya yazın. Duyulan cümle Gemini'ye gider, EMA cevabı okur.</p>
            )}
            {turns.map((turn, index) => (
              <div key={`${turn.role}-${index}`} className={turn.role === 'user' ? 'bubble user' : 'bubble assistant'}>
                <small>{turn.role === 'user' ? 'Sen' : 'EMA'}</small>
                <p>{turn.text}</p>
              </div>
            ))}
          </div>
          <button
            className={recording ? 'speak live' : 'speak'}
            type="button"
            disabled={!ready || !sttReady || busy || hearing}
            onClick={() => (recording ? stopAndReply() : startListening())}
          >
            {recording ? 'Bitir' : hearing ? 'Yazılıyor…' : busy ? 'Cevap hazırlanıyor…' : 'Konuş'}
          </button>
          <label className="check">
            <input
              type="checkbox"
              checked={follow}
              onChange={(event) => setFollow(event.target.checked)}
            />
            Cevaptan sonra tekrar dinle
          </label>
          <form className="composer" onSubmit={sendTyped}>
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Yazarak da deneyebilirsiniz"
              disabled={busy || hearing}
            />
            <button type="submit" disabled={!ready || busy || hearing || !draft.trim()}>Gönder</button>
          </form>
          <p className="error">{error}</p>
          {audioUrl && (
            <div className="player">
              <audio ref={audioRef} src={audioUrl} controls onEnded={onPlaybackEnded} />
              {info && (
                <div className="meta">
                  {info.sttMs != null ? `${Number(info.sttMs).toFixed(0)} ms tanıma · ` : ''}
                  {Number(info.llmMs).toFixed(0)} ms cevap · {Number(info.generateMs).toFixed(0)} ms ses · {Number(info.duration).toFixed(2)} sn
                </div>
              )}
            </div>
          )}
        </section>

        <aside className="panel">
          <label className="block">Ses</label>
          <div className="voice">
            <strong>EMA</strong>
            <small>Cevabı {model} yazar.</small>
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
          <button className="ghost reset" type="button" onClick={resetChat}>Sohbeti sil</button>
        </aside>
      </div>
    </>
  )
}
