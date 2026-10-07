import { useEffect, useRef, useState } from 'react'
import { Room, RoomEvent, Track } from 'livekit-client'

const STATES = {
  initializing: 'Hazırlanıyor',
  idle: 'Bekliyor',
  listening: 'Dinliyor',
  thinking: 'Düşünüyor',
  speaking: 'Konuşuyor',
}

const ACTIONS = {
  ignore: 'Yok sayıldı',
  interrupt: 'Kesildi',
  judge: "Jev'e soruldu",
  blocked: 'Kesmedi',
  warn: 'Uyarı',
  pass: "Jev'e gitmedi",
  nudge: 'Tekrarladı',
}

const PATHS = { jev: "Jev'den geçti", kural: "Jev'e gitmedi" }

export default function Live() {
  const [status, setStatus] = useState('off')
  const [agentState, setAgentState] = useState('')
  const [error, setError] = useState('')
  const [lines, setLines] = useState([])
  const [events, setEvents] = useState([])
  const [judge, setJudge] = useState(null)
  const roomRef = useRef(null)
  const audioRef = useRef(null)
  const listRef = useRef(null)

  useEffect(() => () => roomRef.current?.disconnect(), [])

  useEffect(() => {
    fetch('/api/health')
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => body && setJudge(body))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [lines])

  function upsertLine(id, role, text, final) {
    setLines((current) => {
      const index = current.findIndex((line) => line.id === id)
      const next = { id, role, text, final }
      if (index === -1) return [...current, next]
      const copy = current.slice()
      copy[index] = next
      return copy
    })
  }

  async function connect() {
    setError('')
    setStatus('connecting')
    setLines([])
    setEvents([])
    try {
      const response = await fetch('/api/token')
      if (!response.ok) throw new Error('Oda bilgisi alınamadı.')
      const { url, token } = await response.json()
      const room = new Room({
        audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      roomRef.current = room

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return
        const element = track.attach()
        audioRef.current?.appendChild(element)
      })
      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach().forEach((element) => element.remove())
      })
      room.on(RoomEvent.ParticipantAttributesChanged, (changed) => {
        if (changed['lk.agent.state']) setAgentState(changed['lk.agent.state'])
      })
      room.on(RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
        if (topic !== 'gate') return
        const event = JSON.parse(new TextDecoder().decode(payload))
        setEvents((current) => [{ ...event, at: new Date() }, ...current].slice(0, 30))
      })
      room.on(RoomEvent.Disconnected, () => {
        setStatus('off')
        setAgentState('')
      })
      room.registerTextStreamHandler('lk.transcription', async (reader, participant) => {
        const id = reader.info.attributes?.['lk.segment_id'] || reader.info.id
        const final = reader.info.attributes?.['lk.transcription_final'] === 'true'
        const role = participant.identity === room.localParticipant.identity ? 'user' : 'assistant'
        let text = ''
        for await (const chunk of reader) {
          text = role === 'user' ? chunk : text + chunk
          upsertLine(id, role, text, final)
        }
      })

      await room.connect(url, token)
      await room.startAudio()
      await room.localParticipant.setMicrophoneEnabled(true)
      setStatus('on')
    } catch (err) {
      setError(err.message || 'Bağlanılamadı.')
      setStatus('off')
      roomRef.current?.disconnect()
    }
  }

  async function disconnect() {
    await roomRef.current?.disconnect()
    roomRef.current = null
    setStatus('off')
  }

  return (
    <div className="layout">
      <section className="panel">
        <div className="transcript" ref={listRef}>
          {lines.length === 0 && (
            <p className="empty">
              Görüşmeyi başlatın ve konuşun. EMA konuşurken araya girebilirsiniz; “hıhı”, “tamam” gibi onaylar onu susturmaz.
            </p>
          )}
          {lines.map((line) => (
            <div key={line.id} className={line.role === 'user' ? 'bubble user' : 'bubble assistant'}>
              <small>{line.role === 'user' ? 'Sen' : 'EMA'}</small>
              <p>{line.text}</p>
            </div>
          ))}
        </div>
        <button
          className={status === 'on' ? 'speak live' : 'speak'}
          type="button"
          disabled={status === 'connecting'}
          onClick={() => (status === 'on' ? disconnect() : connect())}
        >
          {status === 'on' ? 'Görüşmeyi bitir' : status === 'connecting' ? 'Bağlanıyor…' : 'Görüşmeyi başlat'}
        </button>
        <p className="error">{error}</p>
        <div ref={audioRef} hidden />
      </section>

      <aside className="panel">
        <label className="block">Ajan</label>
        <div className="voice">
          <strong>{status === 'on' ? STATES[agentState] || 'Bağlandı' : 'Kapalı'}</strong>
          <small>Silero VAD · Zipformer · Gemini · EMA</small>
          {judge?.judge === 'jev' && (
            <small>
              Hakem Jev · {judge.judge_model}
              <br />
              OpenRouter anahtarı <span className="key">{judge.judge_key}</span>
            </small>
          )}
          {judge?.judge === 'gemini' && <small>Hakem Gemini · {judge.judge_model}</small>}
        </div>
        <label className="block">Jev ve kararlar</label>
        <ul className="events">
          {events.length === 0 && <li className="muted">Henüz yok</li>}
          {events.map((event, index) => (
            <li key={`${event.at.getTime()}-${index}`} className={`event ${event.action}`}>
              <b>
                {ACTIONS[event.action] || event.action}
                {event.via && event.action !== 'pass' && (
                  <em className={event.via}>{PATHS[event.via]}</em>
                )}
              </b>
              <span>{event.reason}{event.text ? ` · “${event.text}”` : ''}</span>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  )
}
