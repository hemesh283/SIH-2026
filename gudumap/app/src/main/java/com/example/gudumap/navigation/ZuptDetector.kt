package com.example.gudumap.navigation

import com.example.gudumap.sensor.ImuSample
import kotlin.math.sqrt

enum class NavMotionState {
    STATIONARY,
    ROTATING_IN_PLACE,
    MOVING
}

/**
 * Multi-signal Stationary & In-Place Rotation Detector for Zero-Velocity Updates (ZUPT).
 *
 * Combines:
 * 1. Accelerometer 3D magnitude and variance statistics
 * 2. Horizontal translational acceleration magnitude and variance
 * 3. Gyroscope magnitude and variance statistics
 * 4. Optional GNSS speed
 *
 * Identifies:
 * - STATIONARY: Phone lying completely still on table or resting still in hand.
 * - ROTATING_IN_PLACE: Phone tilted, turned, or handled in place (gyro active, but no horizontal translation).
 * - MOVING: Genuine pedestrian walking or vehicular locomotion.
 *
 * Modular: Can be enabled/disabled dynamically.
 * Configurable: All detection thresholds can be tuned at runtime.
 */
class ZuptDetector(
    var isEnabled: Boolean = true,
    var accMagnitudeThreshold: Float = 0.25f,         // m/s^2 (linear acceleration)
    var accVarianceThreshold: Float = 0.04f,           // (m/s^2)^2
    var horizontalAccThreshold: Float = 0.35f,        // m/s^2 (horizontal translational acceleration)
    var horizontalAccVarianceThreshold: Float = 0.05f,// (m/s^2)^2
    var gyroMagnitudeThreshold: Float = 0.10f,         // rad/s
    var gyroVarianceThreshold: Float = 0.01f,          // (rad/s)^2
    var gnssSpeedThreshold: Float = 0.30f,             // m/s
    var minConsecutiveSamples: Int = 4,                // consecutive stationary samples (~400ms at 10 Hz)
    private val historyWindowSize: Int = 20
) {

    private val accMagHistory = FloatArray(historyWindowSize)
    private val accHorizHistory = FloatArray(historyWindowSize)
    private val gyroMagHistory = FloatArray(historyWindowSize)
    private var historyCount = 0
    private var historyIndex = 0

    private var consecutiveStationaryCount = 0
    private var consecutiveRotatingCount = 0
    private var isStationaryState = false

    var motionState: NavMotionState = NavMotionState.STATIONARY
        private set

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
            motionState = NavMotionState.MOVING
            consecutiveStationaryCount = 0
            consecutiveRotatingCount = 0
            return false
        }

        val accMag = sqrt(sample.ax * sample.ax + sample.ay * sample.ay + sample.az * sample.az)
        val accHoriz = sqrt(sample.ax * sample.ax + sample.ay * sample.ay)
        val gyroMag = sqrt(sample.gx * sample.gx + sample.gy * sample.gy + sample.gz * sample.gz)

        // Store into histories
        accMagHistory[historyIndex] = accMag
        accHorizHistory[historyIndex] = accHoriz
        gyroMagHistory[historyIndex] = gyroMag
        historyIndex = (historyIndex + 1) % historyWindowSize
        if (historyCount < historyWindowSize) {
            historyCount++
        }

        // Calculate variances
        val accVar = calculateVariance(accMagHistory, historyCount)
        val accHorizVar = calculateVariance(accHorizHistory, historyCount)
        val gyroVar = calculateVariance(gyroMagHistory, historyCount)

        // 1. Evaluate full stationary conditions (table, or still in hand)
        val accCondition = (accMag < accMagnitudeThreshold) && (accVar < accVarianceThreshold)
        val gyroCondition = (gyroMag < gyroMagnitudeThreshold) && (gyroVar < gyroVarianceThreshold)
        val gnssCondition = gnssSpeed?.let { it < gnssSpeedThreshold } ?: true

        val instantStationary = accCondition && gyroCondition && gnssCondition

        // 2. Evaluate in-place rotation / handling without translational locomotion
        // Condition: gyro is active (turning, tilting), horizontal acceleration is low and steady, GNSS speed indicates no travel
        val gyroActive = (gyroMag >= gyroMagnitudeThreshold) || (gyroVar >= gyroVarianceThreshold)
        val horizAccLow = (accHoriz < horizontalAccThreshold) && (accHorizVar < horizontalAccVarianceThreshold)
        val instantRotatingInPlace = gyroActive && horizAccLow && gnssCondition

        if (instantStationary) {
            consecutiveStationaryCount++
            consecutiveRotatingCount = 0
            if (consecutiveStationaryCount >= minConsecutiveSamples) {
                motionState = NavMotionState.STATIONARY
                isStationaryState = true
            }
        } else if (instantRotatingInPlace) {
            consecutiveRotatingCount++
            consecutiveStationaryCount = 0
            if (consecutiveRotatingCount >= 2) {
                motionState = NavMotionState.ROTATING_IN_PLACE
                isStationaryState = false
            }
        } else {
            consecutiveStationaryCount = 0
            consecutiveRotatingCount = 0
            motionState = NavMotionState.MOVING
            isStationaryState = false
        }

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
        get() = isEnabled && (motionState == NavMotionState.STATIONARY)

    val isRotatingInPlace: Boolean
        get() = isEnabled && (motionState == NavMotionState.ROTATING_IN_PLACE)

    val isNavStationary: Boolean
        get() = isEnabled && (motionState != NavMotionState.MOVING)

    fun reset() {
        historyCount = 0
        historyIndex = 0
        consecutiveStationaryCount = 0
        consecutiveRotatingCount = 0
        isStationaryState = false
        motionState = NavMotionState.STATIONARY
    }
}
