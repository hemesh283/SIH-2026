package com.example.gudumap.navigation

import com.example.gudumap.sensor.ImuSample
import kotlin.math.sqrt

/**
 * Multi-signal Stationary Detector for Zero-Velocity Updates (ZUPT).
 *
 * Combines:
 * 1. Accelerometer magnitude and variance statistics
 * 2. Gyroscope magnitude and variance statistics
 * 3. Optional GNSS speed
 *
 * Modular: Can be enabled/disabled dynamically.
 * Configurable: All detection thresholds can be tuned at runtime.
 */
class ZuptDetector(
    var isEnabled: Boolean = true,
    var accMagnitudeThreshold: Float = 0.25f,   // m/s^2 (linear acceleration)
    var accVarianceThreshold: Float = 0.04f,     // (m/s^2)^2
    var gyroMagnitudeThreshold: Float = 0.10f,   // rad/s
    var gyroVarianceThreshold: Float = 0.01f,    // (rad/s)^2
    var gnssSpeedThreshold: Float = 0.30f,       // m/s
    var minConsecutiveSamples: Int = 15,        // consecutive stationary samples (~150ms)
    private val historyWindowSize: Int = 20
) {

    private val accMagHistory = FloatArray(historyWindowSize)
    private val gyroMagHistory = FloatArray(historyWindowSize)
    private var historyCount = 0
    private var historyIndex = 0

    private var consecutiveStationaryCount = 0
    private var isStationaryState = false

    /**
     * Feed an incoming IMU sample and optional GNSS speed to update stationary state.
     *
     * @param sample Resampled or raw ImuSample
     * @param gnssSpeed Optional GNSS speed in m/s (null if not available)
     * @return true if stationary conditions are met and detector is enabled
     */
    fun update(sample: ImuSample, gnssSpeed: Float? = null): Boolean {
        if (!isEnabled) {
            isStationaryState = false
            consecutiveStationaryCount = 0
            return false
        }

        val accMag = sqrt(sample.ax * sample.ax + sample.ay * sample.ay + sample.az * sample.az)
        val gyroMag = sqrt(sample.gx * sample.gx + sample.gy * sample.gy + sample.gz * sample.gz)

        // Store into history
        accMagHistory[historyIndex] = accMag
        gyroMagHistory[historyIndex] = gyroMag
        historyIndex = (historyIndex + 1) % historyWindowSize
        if (historyCount < historyWindowSize) {
            historyCount++
        }

        // Calculate variances
        val accVar = calculateVariance(accMagHistory, historyCount)
        val gyroVar = calculateVariance(gyroMagHistory, historyCount)

        // Evaluate multi-signal conditions
        val accCondition = (accMag < accMagnitudeThreshold) && (accVar < accVarianceThreshold)
        val gyroCondition = (gyroMag < gyroMagnitudeThreshold) && (gyroVar < gyroVarianceThreshold)
        val gnssCondition = gnssSpeed?.let { it < gnssSpeedThreshold } ?: true

        val instantStationary = accCondition && gyroCondition && gnssCondition

        if (instantStationary) {
            consecutiveStationaryCount++
        } else {
            consecutiveStationaryCount = 0
        }

        isStationaryState = consecutiveStationaryCount >= minConsecutiveSamples
        return isStationaryState
    }

    private fun calculateVariance(history: FloatArray, count: Int): Float {
        if (count < 2) return 0f
        var sum = 0f
        for (i in 0 until count) {
            sum += history[i]
        }
        val mean = sum / count
        var varSum = 0f
        for (i in 0 until count) {
            val diff = history[i] - mean
            varSum += diff * diff
        }
        return varSum / (count - 1)
    }

    val isStationary: Boolean
        get() = isEnabled && isStationaryState

    fun reset() {
        historyCount = 0
        historyIndex = 0
        consecutiveStationaryCount = 0
        isStationaryState = false
    }
}
