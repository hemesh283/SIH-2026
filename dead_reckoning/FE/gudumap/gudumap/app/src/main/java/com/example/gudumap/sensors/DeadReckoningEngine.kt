package com.example.gudumap.sensors

import android.location.Location
import com.example.gudumap.navigation.DeadReckoningEngine as NavigationEngine
import com.example.gudumap.sensor.ImuSample
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class DeadReckoningState(
    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val speed: Float = 0f,
    val heading: Float = 0f,
    val distanceTravelled: Double = 0.0,
    val isInitialized: Boolean = false
)

/**
 * Backward-compatible bridge adapter for DeadReckoningEngine.
 *
 * Preserves the exact interface expected by NavigationScreen.kt while
 * bridging to the modular com.example.gudumap.navigation.DeadReckoningEngine.
 */
class DeadReckoningEngine(
    val navEngine: NavigationEngine = NavigationEngine()
) {

    companion object {
        private const val EARTH_RADIUS = 6_371_000.0
        private const val MIN_DT = 0.001
        private const val MAX_DT = 0.2
        private const val ACCELERATION_THRESHOLD = 0.12
        private const val VELOCITY_DAMPING = 0.995
    }

    private var latitude = 0.0
    private var longitude = 0.0
    private var velocityNorth = 0.0
    private var velocityEast = 0.0
    private var lastTimestamp = 0L
    private var totalDistance = 0.0
    private var initialized = false

    /**
     * Initialize ONNX Runtime session and assets via underlying navEngine.
     */
    fun initializeAssets(context: android.content.Context) {
        navEngine.initializeAssets(context)
    }

    /**
     * Forward sensor sample to underlying navEngine.
     */
    fun addSensorSample(sample: ImuSample) {
        navEngine.addSensorSample(sample)
    }

    /**
     * Start DR from known geodetic coordinates.
     */
    fun initialize(
        latitude: Double,
        longitude: Double,
        speedMps: Float? = null,
        bearingDeg: Float? = null,
        altitude: Double = 0.0
    ) {
        this.latitude = latitude
        this.longitude = longitude

        if (speedMps != null && bearingDeg != null) {
            val speed = speedMps.toDouble()
            val bearing = Math.toRadians(bearingDeg.toDouble())
            velocityNorth = speed * cos(bearing)
            velocityEast = speed * sin(bearing)
        } else {
            velocityNorth = 0.0
            velocityEast = 0.0
        }

        totalDistance = 0.0
        lastTimestamp = System.nanoTime()
        initialized = true

        // Delegate to modular navigation engine
        navEngine.initialize(latitude, longitude, speedMps, bearingDeg, altitude)
    }

    /**
     * Start DR from a known GNSS position.
     */
    fun initialize(location: Location) {
        val speed = if (location.hasSpeed()) location.speed else null
        val bearing = if (location.hasBearing()) location.bearing else null
        initialize(location.latitude, location.longitude, speed, bearing, location.altitude)
    }

    /**
     * Update DR using world-frame acceleration.
     */
    fun update(
        accelerationNorth: Float,
        accelerationEast: Float,
        heading: Float
    ): DeadReckoningState {
        if (!initialized) {
            return getState()
        }

        val now = System.nanoTime()
        var dt = (now - lastTimestamp) / 1_000_000_000.0
        if (dt < MIN_DT) {
            return getState()
        }
        if (dt > MAX_DT) {
            dt = MAX_DT
        }
        lastTimestamp = now

        // Remove tiny acceleration noise
        val northAcceleration = if (abs(accelerationNorth.toDouble()) < ACCELERATION_THRESHOLD) 0.0 else accelerationNorth.toDouble()
        val eastAcceleration = if (abs(accelerationEast.toDouble()) < ACCELERATION_THRESHOLD) 0.0 else accelerationEast.toDouble()

        velocityNorth += northAcceleration * dt
        velocityEast += eastAcceleration * dt

        velocityNorth *= VELOCITY_DAMPING
        velocityEast *= VELOCITY_DAMPING

        if (abs(velocityNorth) < 0.03 && abs(velocityEast) < 0.03) {
            velocityNorth = 0.0
            velocityEast = 0.0
        }

        val displacementNorth = velocityNorth * dt
        val displacementEast = velocityEast * dt
        val displacement = sqrt(displacementNorth * displacementNorth + displacementEast * displacementEast)
        totalDistance += displacement

        latitude += Math.toDegrees(displacementNorth / EARTH_RADIUS)
        val latitudeRadians = Math.toRadians(latitude)
        val longitudeScale = EARTH_RADIUS * cos(latitudeRadians)
        if (abs(longitudeScale) > 0.000001) {
            longitude += Math.toDegrees(displacementEast / longitudeScale)
        }

        // Also feed into underlying navEngine if samples are converted
        val sample = ImuSample(
            timestampNs = now,
            ax = accelerationNorth,
            ay = accelerationEast,
            az = 0f,
            gx = 0f,
            gy = 0f,
            gz = 0f
        )
        navEngine.addSensorSample(sample)

        return getState(heading)
    }

    /**
     * Correct DR using geodetic coordinates.
     */
    fun correctWithGnss(
        latitude: Double,
        longitude: Double,
        speedMps: Float? = null,
        bearingDeg: Float? = null
    ) {
        if (!initialized) {
            initialize(latitude, longitude, speedMps, bearingDeg)
            return
        }

        this.latitude = latitude
        this.longitude = longitude

        if (speedMps != null && bearingDeg != null) {
            val speed = speedMps.toDouble()
            val bearing = Math.toRadians(bearingDeg.toDouble())
            velocityNorth = speed * cos(bearing)
            velocityEast = speed * sin(bearing)
        }

        lastTimestamp = System.nanoTime()

        // Delegate to modular navigation engine
        navEngine.correctWithGnss(latitude, longitude, speedMps, bearingDeg)
    }

    /**
     * Correct DR using GNSS Location.
     */
    fun correctWithGnss(location: Location) {
        val speed = if (location.hasSpeed()) location.speed else null
        val bearing = if (location.hasBearing()) location.bearing else null
        correctWithGnss(location.latitude, location.longitude, speed, bearing)
    }

    private fun getState(suppliedHeading: Float? = null): DeadReckoningState {
        val speed = sqrt(velocityNorth * velocityNorth + velocityEast * velocityEast)
        val calculatedHeading = if (speed > 0.1) {
            var value = Math.toDegrees(atan2(velocityEast, velocityNorth)).toFloat()
            if (value < 0f) {
                value += 360f
            }
            value
        } else {
            suppliedHeading ?: 0f
        }

        return DeadReckoningState(
            latitude = latitude,
            longitude = longitude,
            speed = speed.toFloat(),
            heading = calculatedHeading,
            distanceTravelled = totalDistance,
            isInitialized = initialized
        )
    }

    fun getState(): DeadReckoningState {
        return getState(null)
    }

    fun reset() {
        latitude = 0.0
        longitude = 0.0
        velocityNorth = 0.0
        velocityEast = 0.0
        lastTimestamp = 0L
        totalDistance = 0.0
        initialized = false
        navEngine.reset()
    }
}