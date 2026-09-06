package com.example.gudumap.sensor

/**
 * Diagnostic telemetry record for device testing and validation (Checkpoint 2 & 3).
 */
data class DiagnosticTelemetryFrame(
    val timestampNs: Long,
    val accX: Float,
    val accY: Float,
    val accZ: Float,
    val gyroX: Float,
    val gyroY: Float,
    val gyroZ: Float,
    val effectiveSampleIntervalMs: Float,
    val bufferSize: Int,
    val mlInferenceTimestampNs: Long,
    val mlInferenceLatencyMs: Float,
    val dxMeters: Float,
    val dyMeters: Float,
    val dzMeters: Float,
    val headingDeg: Float,
    val isGnssAvailable: Boolean,
    val isBlackout: Boolean
)

/**
 * Lightweight, thread-safe diagnostic telemetry recorder for dead reckoning device testing.
 * Designed to incur near-zero overhead when disabled (production default).
 */
class DiagnosticRecorder(
    private val maxCapacity: Int = 1000
) {
    @Volatile
    var isEnabled: Boolean = false

    private val lock = Any()
    private val frames = ArrayDeque<DiagnosticTelemetryFrame>(maxCapacity)
    private var lastSampleTimestampNs: Long = 0L

    fun recordSample(
        sample: ImuSample,
        bufferSize: Int,
        headingDeg: Float,
        isGnssAvailable: Boolean,
        isBlackout: Boolean,
        lastInferenceTimestampNs: Long = 0L,
        lastInferenceLatencyMs: Float = 0f,
        lastDisplacement: FloatArray = floatArrayOf(0f, 0f, 0f)
    ) {
        if (!isEnabled) return

        val dtMs = if (lastSampleTimestampNs > 0L) {
            (sample.timestampNs - lastSampleTimestampNs) / 1_000_000f
        } else 0f
        lastSampleTimestampNs = sample.timestampNs

        val frame = DiagnosticTelemetryFrame(
            timestampNs = sample.timestampNs,
            accX = sample.ax,
            accY = sample.ay,
            accZ = sample.az,
            gyroX = sample.gx,
            gyroY = sample.gy,
            gyroZ = sample.gz,
            effectiveSampleIntervalMs = dtMs,
            bufferSize = bufferSize,
            mlInferenceTimestampNs = lastInferenceTimestampNs,
            mlInferenceLatencyMs = lastInferenceLatencyMs,
            dxMeters = if (lastDisplacement.isNotEmpty()) lastDisplacement[0] else 0f,
            dyMeters = if (lastDisplacement.size > 1) lastDisplacement[1] else 0f,
            dzMeters = if (lastDisplacement.size > 2) lastDisplacement[2] else 0f,
            headingDeg = headingDeg,
            isGnssAvailable = isGnssAvailable,
            isBlackout = isBlackout
        )

        synchronized(lock) {
            if (frames.size >= maxCapacity) {
                frames.removeFirst()
            }
            frames.addLast(frame)
        }
    }

    fun getRecentFrames(): List<DiagnosticTelemetryFrame> {
        return synchronized(lock) { frames.toList() }
    }

    fun clear() {
        synchronized(lock) {
            frames.clear()
            lastSampleTimestampNs = 0L
        }
    }

    val frameCount: Int
        get() = synchronized(lock) { frames.size }
}
