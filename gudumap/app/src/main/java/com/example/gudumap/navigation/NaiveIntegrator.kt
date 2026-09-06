package com.example.gudumap.navigation

/**
 * Pure double-integration of world-frame accelerometer samples -- no ZUPT, no ML
 * displacement correction, no EKF fusion, no measurement updates of any kind.
 *
 * Exists solely to render alongside the corrected GRU+EKF+ZUPT trajectory as a visual
 * contrast: this is what raw phone-grade MEMS dead reckoning looks like on its own,
 * which is the baseline the rest of this project is built to fix.
 */
class NaiveIntegrator {

    private var velocityNorth = 0.0
    private var velocityEast = 0.0

    var displacementNorth = 0.0
        private set
    var displacementEast = 0.0
        private set

    private var lastSampleNs: Long = 0L
    private var hasLastSample = false

    fun reset() {
        velocityNorth = 0.0
        velocityEast = 0.0
        displacementNorth = 0.0
        displacementEast = 0.0
        hasLastSample = false
    }

    /**
     * @param accelNorth world-frame (NED) North acceleration, m/s^2, gravity already removed
     * @param accelEast world-frame (NED) East acceleration, m/s^2, gravity already removed
     */
    fun addSample(timestampNs: Long, accelNorth: Float, accelEast: Float) {
        if (!hasLastSample) {
            lastSampleNs = timestampNs
            hasLastSample = true
            return
        }
        val dt = (timestampNs - lastSampleNs) / 1_000_000_000.0
        lastSampleNs = timestampNs
        // Guard against clock resets or long gaps producing a single huge integration step
        if (dt <= 0.0 || dt > 1.0) return

        velocityNorth += accelNorth * dt
        velocityEast += accelEast * dt
        displacementNorth += velocityNorth * dt
        displacementEast += velocityEast * dt
    }
}
