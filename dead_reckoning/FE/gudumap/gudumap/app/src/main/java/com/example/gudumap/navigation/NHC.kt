package com.example.gudumap.navigation

import kotlin.math.cos
import kotlin.math.sin

/**
 * Non-Holonomic Constraints (NHC) for wheeled vehicle dead reckoning.
 *
 * In land vehicles under normal driving conditions without sideslip or flight:
 * - Lateral velocity in vehicle body frame: v_y^b ~ 0
 * - Vertical velocity in vehicle body frame: v_z^b ~ 0
 *
 * Applied as an EKF pseudo-measurement update.
 * Modular: Can be enabled/disabled dynamically.
 */
class NHC(
    var isEnabled: Boolean = true,
    var lateralNoiseMps: Double = 0.15,   // Lateral velocity measurement noise
    var verticalNoiseMps: Double = 0.10   // Vertical velocity measurement noise
) {

    /**
     * Apply NHC measurement update to the EKF.
     *
     * @param ekf Active Extended Kalman Filter
     * @param headingDeg Vehicle heading in degrees (clockwise from North)
     * @param pitchDeg Optional vehicle pitch in degrees (default 0)
     */
    fun applyConstraint(ekf: EKF, headingDeg: Float, pitchDeg: Float = 0f) {
        if (!isEnabled || !ekf.isInitialized) return

        val headingRad = Math.toRadians(headingDeg.toDouble())
        val cosH = cos(headingRad)
        val sinH = sin(headingRad)

        // For 2D / planar motion:
        // Body frame: X is forward, Y is right (lateral), Z is down.
        // v_forward = cos(H)*v_N + sin(H)*v_E
        // v_lateral = -sin(H)*v_N + cos(H)*v_E ~ 0
        // v_vertical = v_D ~ 0

        // Pseudo-measurement z = [0, 0]
        val z = doubleArrayOf(0.0, 0.0)

        // Measurement matrix H (2 x 6) relating [p_N, p_E, p_D, v_N, v_E, v_D]
        val H = arrayOf(
            // v_lateral row
            doubleArrayOf(0.0, 0.0, 0.0, -sinH, cosH, 0.0),
            // v_vertical row
            doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        )

        val rLat = lateralNoiseMps * lateralNoiseMps
        val rVert = verticalNoiseMps * verticalNoiseMps
        val R = arrayOf(
            doubleArrayOf(rLat, 0.0),
            doubleArrayOf(0.0, rVert)
        )

        ekf.updateMeasurement(z, H, R)
    }
}
