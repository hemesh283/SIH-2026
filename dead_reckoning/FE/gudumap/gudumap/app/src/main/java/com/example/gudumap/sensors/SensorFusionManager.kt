package com.example.gudumap.sensors

import android.hardware.SensorManager
import kotlin.math.sqrt

data class OrientationData(
    val heading: Float = 0f,
    val pitch: Float = 0f,
    val roll: Float = 0f
)

data class WorldAcceleration(
    val north: Float = 0f,
    val east: Float = 0f,
    val vertical: Float = 0f
)

class SensorFusionManager {

    private val accelerometer = FloatArray(3)
    private val magnetometer = FloatArray(3)
    private val gyroscope = FloatArray(3)

    private val rotationMatrix = FloatArray(9)
    private val inclinationMatrix = FloatArray(9)
    private val orientation = FloatArray(3)

    private var hasAccelerometer = false
    private var hasMagnetometer = false
    private var hasGyroscope = false

    // Fused orientation in radians
    private var fusedAzimuth = 0f
    private var fusedPitch = 0f
    private var fusedRoll = 0f

    private var lastGyroTimestamp = 0L

    // Complementary filter weights
    private val gyroWeight = 0.98f
    private val sensorWeight = 0.02f

    fun updateAccelerometer(
        x: Float,
        y: Float,
        z: Float
    ) {
        accelerometer[0] = x
        accelerometer[1] = y
        accelerometer[2] = z

        hasAccelerometer = true

        updateSensorOrientation()
    }

    fun updateMagnetometer(
        x: Float,
        y: Float,
        z: Float
    ) {
        magnetometer[0] = x
        magnetometer[1] = y
        magnetometer[2] = z

        hasMagnetometer = true

        updateSensorOrientation()
    }

    fun updateGyroscope(
        x: Float,
        y: Float,
        z: Float
    ) {

        gyroscope[0] = x
        gyroscope[1] = y
        gyroscope[2] = z

        hasGyroscope = true

        val currentTime = System.nanoTime()

        if (lastGyroTimestamp == 0L) {
            lastGyroTimestamp = currentTime
            return
        }

        val dt =
            (currentTime - lastGyroTimestamp) /
                    1_000_000_000f

        lastGyroTimestamp = currentTime

        /*
         * Ignore invalid time intervals.
         */
        if (dt <= 0f || dt > 0.2f) {
            return
        }

        /*
         * Integrate gyro angular velocity.
         *
         * Gyroscope values are radians/second.
         */
        fusedAzimuth += gyroscope[2] * dt
        fusedPitch += gyroscope[1] * dt
        fusedRoll += gyroscope[0] * dt

        fusedAzimuth =
            normalizeAngle(fusedAzimuth)

        /*
         * Correct gyro drift using
         * accelerometer + magnetometer.
         */
        if (
            hasAccelerometer &&
            hasMagnetometer
        ) {

            val success =
                SensorManager.getRotationMatrix(
                    rotationMatrix,
                    inclinationMatrix,
                    accelerometer,
                    magnetometer
                )

            if (success) {

                SensorManager.getOrientation(
                    rotationMatrix,
                    orientation
                )

                fusedAzimuth =
                    complementaryFilter(
                        fusedAzimuth,
                        orientation[0]
                    )

                fusedPitch =
                    complementaryFilter(
                        fusedPitch,
                        orientation[1]
                    )

                fusedRoll =
                    complementaryFilter(
                        fusedRoll,
                        orientation[2]
                    )
            }
        }
    }

    /*
     * Calculate orientation from
     * accelerometer + magnetometer.
     */
    private fun updateSensorOrientation() {

        if (
            !hasAccelerometer ||
            !hasMagnetometer
        ) {
            return
        }

        val success =
            SensorManager.getRotationMatrix(
                rotationMatrix,
                inclinationMatrix,
                accelerometer,
                magnetometer
            )

        if (!success) {
            return
        }

        SensorManager.getOrientation(
            rotationMatrix,
            orientation
        )

        /*
         * Before gyro data arrives,
         * use the sensor orientation directly.
         */
        if (!hasGyroscope) {

            fusedAzimuth =
                orientation[0]

            fusedPitch =
                orientation[1]

            fusedRoll =
                orientation[2]
        }
    }

    /*
     * Complementary filter.
     */
    private fun complementaryFilter(
        gyroAngle: Float,
        sensorAngle: Float
    ): Float {

        var difference =
            sensorAngle - gyroAngle

        while (difference > Math.PI.toFloat()) {
            difference -=
                (2f * Math.PI.toFloat())
        }

        while (difference < -Math.PI.toFloat()) {
            difference +=
                (2f * Math.PI.toFloat())
        }

        return normalizeAngle(
            gyroAngle * gyroWeight +
                    (
                            gyroAngle +
                                    difference
                            ) * sensorWeight
        )
    }

    /*
     * Keep angle between -PI and PI.
     */
    private fun normalizeAngle(
        angle: Float
    ): Float {

        var result = angle

        while (
            result > Math.PI.toFloat()
        ) {
            result -=
                2f * Math.PI.toFloat()
        }

        while (
            result < -Math.PI.toFloat()
        ) {
            result +=
                2f * Math.PI.toFloat()
        }

        return result
    }

    /*
     * Return fused orientation.
     */
    fun getOrientation(): OrientationData {

        var heading =
            Math.toDegrees(
                fusedAzimuth.toDouble()
            ).toFloat()

        if (heading < 0f) {
            heading += 360f
        }

        return OrientationData(

            heading = heading,

            pitch =
                Math.toDegrees(
                    fusedPitch.toDouble()
                ).toFloat(),

            roll =
                Math.toDegrees(
                    fusedRoll.toDouble()
                ).toFloat()
        )
    }

    /*
     * Convert phone-frame acceleration
     * into world-frame acceleration.
     *
     * X = East
     * Y = North
     * Z = vertical
     */
    fun getWorldAcceleration():
            WorldAcceleration {

        if (
            !hasAccelerometer ||
            !hasMagnetometer
        ) {
            return WorldAcceleration()
        }

        val success =
            SensorManager.getRotationMatrix(
                rotationMatrix,
                inclinationMatrix,
                accelerometer,
                magnetometer
            )

        if (!success) {
            return WorldAcceleration()
        }

        val worldX =
            rotationMatrix[0] * accelerometer[0] +
                    rotationMatrix[1] * accelerometer[1] +
                    rotationMatrix[2] * accelerometer[2]

        val worldY =
            rotationMatrix[3] * accelerometer[0] +
                    rotationMatrix[4] * accelerometer[1] +
                    rotationMatrix[5] * accelerometer[2]

        val worldZ =
            rotationMatrix[6] * accelerometer[0] +
                    rotationMatrix[7] * accelerometer[1] +
                    rotationMatrix[8] * accelerometer[2]

        /*
         * Remove gravity.
         */
        val gravity = 9.81f

        val correctedVertical =
            worldZ - gravity

        return WorldAcceleration(

            north = worldY,

            east = worldX,

            vertical = correctedVertical
        )
    }

    /*
     * Raw accelerometer magnitude.
     */
    fun getAccelerometerMagnitude(): Float {

        return sqrt(
            accelerometer[0] *
                    accelerometer[0] +

                    accelerometer[1] *
                    accelerometer[1] +

                    accelerometer[2] *
                    accelerometer[2]
        )
    }

    /*
     * Raw gyroscope magnitude.
     */
    fun getGyroscopeMagnitude(): Float {

        return sqrt(
            gyroscope[0] *
                    gyroscope[0] +

                    gyroscope[1] *
                    gyroscope[1] +

                    gyroscope[2] *
                    gyroscope[2]
        )
    }

    fun hasGyroscope(): Boolean {
        return hasGyroscope
    }

    fun reset() {

        fusedAzimuth = 0f
        fusedPitch = 0f
        fusedRoll = 0f

        lastGyroTimestamp = 0L

        hasAccelerometer = false
        hasMagnetometer = false
        hasGyroscope = false
    }
}