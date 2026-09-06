package com.example.gudumap.sensors

import android.hardware.SensorManager
import kotlin.math.sqrt

data class OrientationData(
    val heading: Float = 0f, // Azimuth in degrees [0, 360)
    val pitch: Float = 0f,   // Pitch in degrees [-90, 90]
    val roll: Float = 0f     // Roll in degrees [-180, 180]
)

data class WorldAcceleration(
    val north: Float = 0f, // m/s^2 (towards geographic North)
    val east: Float = 0f,  // m/s^2 (towards East)
    val vertical: Float = 0f, // m/s^2 (towards Up)
    val timestampNs: Long = 0L
)

/**
 * How much to trust the current heading estimate. TYPE_ROTATION_VECTOR fuses the
 * magnetometer, and magnetometer reliability is exactly what degrades inside a vehicle
 * chassis or tunnel/parking-garage rebar -- the scenarios this project targets. Previously
 * Android's own SENSOR_STATUS_* reliability signal for these sensors was read and discarded;
 * this surfaces it instead of pretending heading quality is constant.
 */
enum class HeadingConfidence {
    HIGH, MEDIUM, LOW, UNRELIABLE
}

class SensorFusionManager {

    private val linearAccelerometer = FloatArray(3)
    private val magnetometer = FloatArray(3)
    private val gyroscope = FloatArray(3)

    // Current 3x3 rotation matrix from device to world (X=East, Y=North, Z=Up)
    private val rotationMatrix = FloatArray(9) { if (it % 4 == 0) 1f else 0f }
    private val inclinationMatrix = FloatArray(9)
    private val orientationAngles = FloatArray(3)

    private var hasAccelerometer = false
    private var hasMagnetometer = false
    private var hasGyroscope = false
    private var hasHardwareRotation = false

    private var fusedAzimuth = 0f
    private var fusedPitch = 0f
    private var fusedRoll = 0f

    private var lastGyroTimestampNs = 0L
    private var lastAccelTimestampNs = 0L

    private val gyroWeight = 0.98f
    private val sensorWeight = 0.02f

    private var magnetometerAccuracy: Int = SensorManager.SENSOR_STATUS_UNRELIABLE
    private var rotationVectorAccuracy: Int = SensorManager.SENSOR_STATUS_UNRELIABLE

    fun updateMagnetometerAccuracy(accuracy: Int) {
        magnetometerAccuracy = accuracy
    }

    fun updateRotationVectorAccuracy(accuracy: Int) {
        rotationVectorAccuracy = accuracy
    }

    /**
     * Worst-of-the-two reliability across the magnetometer and the rotation-vector sensor
     * that consumes it. Deliberately pessimistic: either sensor being unreliable means the
     * fused heading it feeds into is suspect, regardless of what the other one reports.
     */
    val headingConfidence: HeadingConfidence
        get() {
            val worst = minOf(magnetometerAccuracy, rotationVectorAccuracy)
            return when (worst) {
                SensorManager.SENSOR_STATUS_ACCURACY_HIGH -> HeadingConfidence.HIGH
                SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM -> HeadingConfidence.MEDIUM
                SensorManager.SENSOR_STATUS_ACCURACY_LOW -> HeadingConfidence.LOW
                else -> HeadingConfidence.UNRELIABLE // SENSOR_STATUS_UNRELIABLE, SENSOR_STATUS_NO_CONTACT, or not yet reported
            }
        }

    fun updateAccelerometer(x: Float, y: Float, z: Float, timestampNs: Long = System.nanoTime()) {
        linearAccelerometer[0] = x
        linearAccelerometer[1] = y
        linearAccelerometer[2] = z
        lastAccelTimestampNs = timestampNs
        hasAccelerometer = true

        if (!hasHardwareRotation) {
            updateSensorOrientation()
        }
    }

    fun updateMagnetometer(x: Float, y: Float, z: Float, timestampNs: Long = System.nanoTime()) {
        magnetometer[0] = x
        magnetometer[1] = y
        magnetometer[2] = z
        hasMagnetometer = true

        if (!hasHardwareRotation) {
            updateSensorOrientation()
        }
    }

    fun updateGyroscope(x: Float, y: Float, z: Float, timestampNs: Long = System.nanoTime()) {
        gyroscope[0] = x
        gyroscope[1] = y
        gyroscope[2] = z
        hasGyroscope = true

        if (lastGyroTimestampNs == 0L) {
            lastGyroTimestampNs = timestampNs
            return
        }

        val dt = (timestampNs - lastGyroTimestampNs) / 1_000_000_000f
        lastGyroTimestampNs = timestampNs

        if (dt <= 0f || dt > 0.2f) return

        if (!hasHardwareRotation) {
            fusedAzimuth += gyroscope[2] * dt
            fusedPitch += gyroscope[1] * dt
            fusedRoll += gyroscope[0] * dt

            fusedAzimuth = normalizeAngle(fusedAzimuth)

            if (hasAccelerometer && hasMagnetometer) {
                val success = SensorManager.getRotationMatrix(
                    rotationMatrix,
                    inclinationMatrix,
                    linearAccelerometer,
                    magnetometer
                )
                if (success) {
                    SensorManager.getOrientation(rotationMatrix, orientationAngles)
                    fusedAzimuth = complementaryFilter(fusedAzimuth, orientationAngles[0])
                    fusedPitch = complementaryFilter(fusedPitch, orientationAngles[1])
                    fusedRoll = complementaryFilter(fusedRoll, orientationAngles[2])
                }
            }
        }
    }

    /**
     * Updates rotation directly from Android's hardware-fused Rotation Vector sensor.
     */
    fun updateRotationVector(matrix: FloatArray, timestampNs: Long = System.nanoTime()) {
        if (matrix.size >= 9) {
            System.arraycopy(matrix, 0, rotationMatrix, 0, 9)
            SensorManager.getOrientation(rotationMatrix, orientationAngles)
            fusedAzimuth = orientationAngles[0]
            fusedPitch = orientationAngles[1]
            fusedRoll = orientationAngles[2]
            hasHardwareRotation = true
        }
    }

    private fun updateSensorOrientation() {
        if (!hasAccelerometer || !hasMagnetometer) return

        val success = SensorManager.getRotationMatrix(
            rotationMatrix,
            inclinationMatrix,
            linearAccelerometer,
            magnetometer
        )

        if (success) {
            SensorManager.getOrientation(rotationMatrix, orientationAngles)
            if (!hasGyroscope) {
                fusedAzimuth = orientationAngles[0]
                fusedPitch = orientationAngles[1]
                fusedRoll = orientationAngles[2]
            }
        }
    }

    private fun complementaryFilter(gyroAngle: Float, sensorAngle: Float): Float {
        var diff = sensorAngle - gyroAngle
        while (diff > Math.PI.toFloat()) diff -= (2f * Math.PI.toFloat())
        while (diff < -Math.PI.toFloat()) diff += (2f * Math.PI.toFloat())

        return normalizeAngle(gyroAngle * gyroWeight + (gyroAngle + diff) * sensorWeight)
    }

    private fun normalizeAngle(angle: Float): Float {
        var result = angle
        while (result > Math.PI.toFloat()) result -= 2f * Math.PI.toFloat()
        while (result < -Math.PI.toFloat()) result += 2f * Math.PI.toFloat()
        return result
    }

    fun getOrientation(): OrientationData {
        var heading = Math.toDegrees(fusedAzimuth.toDouble()).toFloat()
        if (heading < 0f) heading += 360f

        return OrientationData(
            heading = heading,
            pitch = Math.toDegrees(fusedPitch.toDouble()).toFloat(),
            roll = Math.toDegrees(fusedRoll.toDouble()).toFloat()
        )
    }

    fun getRotationMatrix(): FloatArray {
        return rotationMatrix.clone()
    }

    /**
     * Transforms device linear acceleration to World Coordinates:
     * East (X), North (Y), Vertical (Z).
     */
    fun getWorldAcceleration(): WorldAcceleration {
        val ax = linearAccelerometer[0]
        val ay = linearAccelerometer[1]
        val az = linearAccelerometer[2]

        // World coordinates = R * Device coordinates
        val east = rotationMatrix[0] * ax + rotationMatrix[1] * ay + rotationMatrix[2] * az
        val north = rotationMatrix[3] * ax + rotationMatrix[4] * ay + rotationMatrix[5] * az
        val vertical = rotationMatrix[6] * ax + rotationMatrix[7] * ay + rotationMatrix[8] * az

        return WorldAcceleration(
            north = north,
            east = east,
            vertical = vertical,
            timestampNs = lastAccelTimestampNs
        )
    }

    fun reset() {
        fusedAzimuth = 0f
        fusedPitch = 0f
        fusedRoll = 0f
        lastGyroTimestampNs = 0L
        lastAccelTimestampNs = 0L
        hasAccelerometer = false
        hasMagnetometer = false
        hasGyroscope = false
        hasHardwareRotation = false
        magnetometerAccuracy = SensorManager.SENSOR_STATUS_UNRELIABLE
        rotationVectorAccuracy = SensorManager.SENSOR_STATUS_UNRELIABLE
        for (i in 0 until 9) {
            rotationMatrix[i] = if (i % 4 == 0) 1f else 0f
        }
    }
}