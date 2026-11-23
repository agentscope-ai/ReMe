package io.agentscope.reme.studio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Base64
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.abs

internal class AndroidRecorder {
    private class Capture {
        @Volatile var stopping = false
        @Volatile var frames = 0
        @Volatile var level = 0.0
        var finished = false
        var audio: String? = null
        var result: ((String?) -> Unit)? = null
    }

    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private var capture: Capture? = null

    val seconds: Double get() = (capture?.frames ?: 0).toDouble() / SAMPLE_RATE
    val level: Double get() = capture?.level ?: 0.0

    fun start(ready: (Boolean) -> Unit) {
        cancel()
        val next = Capture()
        capture = next
        worker.execute {
            val audio = record(next) { started ->
                main.post { ready(started && capture === next && !next.stopping) }
            }
            main.post {
                if (capture !== next) return@post
                next.finished = true
                next.audio = audio
                next.result?.let { finish ->
                    capture = null
                    finish(audio)
                }
            }
        }
    }

    fun stop(result: (String?) -> Unit) {
        val current = capture
        if (current == null || current.result != null) {
            result(null)
        } else if (current.finished) {
            capture = null
            result(current.audio)
        } else {
            current.result = result
            current.stopping = true
        }
    }

    fun cancel() {
        val current = capture ?: return
        capture = null
        current.stopping = true
        current.result?.invoke(null)
    }

    fun close() {
        cancel()
        worker.shutdown()
    }

    @SuppressLint("MissingPermission") // The activity checks RECORD_AUDIO before calling start.
    private fun record(capture: Capture, ready: (Boolean) -> Unit): String? {
        var recorder: AudioRecord? = null
        var started = false
        try {
            if (capture.stopping) return null
            val minimum = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (minimum <= 0) return null
            val format = AudioFormat.Builder()
                .setSampleRate(SAMPLE_RATE)
                .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .build()
            val input = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.MIC)
                .setAudioFormat(format)
                .setBufferSizeInBytes(maxOf(minimum, SAMPLE_RATE / 5))
                .build()
            recorder = input
            if (input.state != AudioRecord.STATE_INITIALIZED || capture.stopping) return null
            input.startRecording()
            if (input.recordingState != AudioRecord.RECORDSTATE_RECORDING) return null

            val samples = ShortArray(SAMPLE_RATE * MAX_SECONDS)
            val until = SystemClock.elapsedRealtime() + 4000
            while (!capture.stopping && capture.frames < samples.size) {
                val offset = capture.frames
                val count = input.read(samples, offset, minOf(320, samples.size - offset), AudioRecord.READ_NON_BLOCKING)
                if (count < 0) return null
                if (count == 0) {
                    if (!started && SystemClock.elapsedRealtime() >= until) return null
                    SystemClock.sleep(10)
                    continue
                }
                var peak = 0
                for (index in offset until offset + count) peak = maxOf(peak, abs(samples[index].toInt()))
                capture.frames = offset + count
                capture.level = (peak / 6553.6).coerceAtMost(1.0)
                if (!started) {
                    started = true
                    ready(true)
                }
            }
            val pcm = ByteBuffer.allocate(capture.frames * 2).order(ByteOrder.LITTLE_ENDIAN)
            pcm.asShortBuffer().put(samples, 0, capture.frames)
            return Base64.encodeToString(pcm.array(), Base64.NO_WRAP)
        } catch (_: RuntimeException) {
            return null
        } finally {
            if (!started) ready(false)
            recorder?.release()
        }
    }

    private companion object {
        const val SAMPLE_RATE = 16000
        const val MAX_SECONDS = 24
    }
}
