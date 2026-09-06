package com.example.gudumap.navigation

import com.example.gudumap.ml.ModelMetadata
import com.example.gudumap.sensor.ImuSample

/**
 * Timestamp-aware IMU buffer with 10 Hz linear interpolation resampling
 * and sliding window extraction with stride scheduling.
 *
 * Contract (IO-VNBD Native 10 Hz):
 * - Resampling frequency: 10 Hz (100 ms grid)
 * - Window size: 20 samples (2.0 seconds)
 * - Stride: 10 samples (1.0 second between consecutive overlapping windows)
 * - Window shape: [20][6]
 */
class IMUBuffer(
    val windowSize: Int = ModelMetadata.WINDOW_SIZE,
    val stride: Int = ModelMetadata.STRIDE,
    val targetDtNs: Long = 100_000_000L // 10 Hz = 100 ms = 100,000,000 ns
) {

    private val lock = Any()

    // Resampling state
    private var previousSample: ImuSample? = null
    private var nextGridTimestampNs: Long = 0L

    // Fixed 10 Hz sample ring buffer
    // Holds up to (windowSize + stride * 2) samples
    private val resampledBuffer = ArrayList<FloatArray>(windowSize + stride)
    private var samplesSinceLastWindow: Int = 0

    // Callback when a 20x6 window is ready for inference
    var onWindowReadyListener: ((Array<FloatArray>, Long) -> Unit)? = null

    /**
     * Add a new raw IMU sample from sensor callbacks.
     * Android callbacks can have timestamp jitter; this resamples them to exact 10 Hz.
     */
    fun addSample(sample: ImuSample) {
        synchronized(lock) {
            val prev = previousSample
            if (prev == null) {
                previousSample = sample
                nextGridTimestampNs = sample.timestampNs
                return
            }

            // Guard against backward or duplicate timestamps
            if (sample.timestampNs <= prev.timestampNs) {
                return
            }

            // Handle large discontinuities (> 500 ms)
            if (sample.timestampNs - prev.timestampNs > 500_000_000L) {
                previousSample = sample
                nextGridTimestampNs = sample.timestampNs
                return
            }

            // Interpolate all grid points that fall between prev and current sample
            while (nextGridTimestampNs <= sample.timestampNs) {
                if (nextGridTimestampNs >= prev.timestampNs) {
                    val dtTotal = (sample.timestampNs - prev.timestampNs).toFloat()
                    val dtTarget = (nextGridTimestampNs - prev.timestampNs).toFloat()
                    val alpha = if (dtTotal > 0f) dtTarget / dtTotal else 0f

                    val resampled = FloatArray(ModelMetadata.FEATURE_DIM)
                    // acc_x, acc_y, acc_z in g (linear acceleration gravity-removed)
                    resampled[0] = lerp(prev.ax, sample.ax, alpha) / ModelMetadata.GRAVITY_MPS2
                    resampled[1] = lerp(prev.ay, sample.ay, alpha) / ModelMetadata.GRAVITY_MPS2
                    resampled[2] = lerp(prev.az, sample.az, alpha) / ModelMetadata.GRAVITY_MPS2
                    // gyro_x, gyro_y, gyro_z in rad/s
                    resampled[3] = lerp(prev.gx, sample.gx, alpha)
                    resampled[4] = lerp(prev.gy, sample.gy, alpha)
                    resampled[5] = lerp(prev.gz, sample.gz, alpha)

                    appendResampledSample(resampled, nextGridTimestampNs)
                }
                nextGridTimestampNs += targetDtNs
            }

            previousSample = sample
        }
    }

    private fun lerp(a: Float, b: Float, alpha: Float): Float {
        return a + alpha * (b - a)
    }

    private fun appendResampledSample(sample: FloatArray, timestampNs: Long) {
        resampledBuffer.add(sample)
        samplesSinceLastWindow++

        // Check if we have enough samples for a window
        if (resampledBuffer.size >= windowSize) {
            val shouldEmit = if (resampledBuffer.size == windowSize) {
                // First initial window (samples 0..19)
                true
            } else {
                // Subsequent overlapping windows every `stride` samples (e.g. 10 samples)
                samplesSinceLastWindow >= stride
            }

            if (shouldEmit) {
                // Extract slice of the last `windowSize` samples
                val startIndex = resampledBuffer.size - windowSize
                val window = Array(windowSize) { i ->
                    resampledBuffer[startIndex + i].clone()
                }

                samplesSinceLastWindow = 0

                // Trim buffer to avoid unbounded memory growth
                // Keep only enough past samples to form the next window
                val maxKeep = windowSize + stride
                if (resampledBuffer.size > maxKeep) {
                    val removeCount = resampledBuffer.size - windowSize
                    for (i in 0 until removeCount) {
                        resampledBuffer.removeAt(0)
                    }
                }

                // Dispatch window
                onWindowReadyListener?.invoke(window, timestampNs)
            }
        }
    }

    /**
     * Resets all buffer state.
     */
    fun reset() {
        synchronized(lock) {
            previousSample = null
            nextGridTimestampNs = 0L
            resampledBuffer.clear()
            samplesSinceLastWindow = 0
        }
    }

    val size: Int
        get() = synchronized(lock) { resampledBuffer.size }
}
