package com.example.gudumap.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager as AndroidSensorManager
import kotlin.math.sqrt

// =========================================================
// SENSOR DATA CLASSES
// =========================================================

data class AccelerometerData(
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f,
    val timestampNs: Long = 0L,
    val isLinearAcceleration: Boolean = false
) {
    val magnitude: Float
        get() = sqrt(x * x + y * y + z * z)
}

data class GyroscopeData(
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f,
    val timestampNs: Long = 0L
) {
    val magnitude: Float
        get() = sqrt(x * x + y * y + z * z)
}

data class MagnetometerData(
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f,
    val timestampNs: Long = 0L
) {
    val magnitude: Float
        get() = sqrt(x * x + y * y + z * z)
}

data class RotationVectorData(
    val values: FloatArray = FloatArray(4),
    val rotationMatrix: FloatArray = FloatArray(9) { if (it % 4 == 0) 1f else 0f },
    val timestampNs: Long = 0L
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as RotationVectorData
        return values.contentEquals(other.values) && rotationMatrix.contentEquals(other.rotationMatrix)
    }

    override fun hashCode(): Int {
        var result = values.contentHashCode()
        result = 31 * result + rotationMatrix.contentHashCode()
        return result
    }
}

// =========================================================
// SENSOR MANAGER
// =========================================================

class SensorManager(
    context: Context
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as AndroidSensorManager

    // ---------------------------------------------------------
    // SENSORS
    // ---------------------------------------------------------

    // Prefer TYPE_LINEAR_ACCELERATION (gravity removed) as required by ML deployment contract
    private val linearAccelerometer =
        sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

    private val rawAccelerometer =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val gyroscope =
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private val magnetometer =
        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    private val rotationVectorSensor =
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    // Software gravity filter if linear accelerometer is absent
    private val gravityValues = FloatArray(3)
    private var hasGravity = false
    private val gravityAlpha = 0.8f

    // ---------------------------------------------------------
    // SENSOR STATUS FLAGS
    // ---------------------------------------------------------

    var isAccelerometerActive: Boolean = false
        private set

    var isGyroscopeActive: Boolean = false
        private set

    var isMagnetometerActive: Boolean = false
        private set

    var isRotationVectorActive: Boolean = false
        private set

    // Android's own SENSOR_STATUS_* reliability signal for heading-relevant sensors --
    // previously read and discarded; now tracked so heading confidence can be surfaced.
    var magnetometerAccuracy: Int = AndroidSensorManager.SENSOR_STATUS_UNRELIABLE
        private set

    var rotationVectorAccuracy: Int = AndroidSensorManager.SENSOR_STATUS_UNRELIABLE
        private set

    // ---------------------------------------------------------
    // CALLBACKS
    // ---------------------------------------------------------

    private var onAccelerometerChanged: ((AccelerometerData) -> Unit)? = null
    private var onGyroscopeChanged: ((GyroscopeData) -> Unit)? = null
    private var onMagnetometerChanged: ((MagnetometerData) -> Unit)? = null
    private var onRotationVectorChanged: ((RotationVectorData) -> Unit)? = null

    // =========================================================
    // ACCELEROMETER
    // =========================================================

    fun startAccelerometer(callback: (AccelerometerData) -> Unit) {
        onAccelerometerChanged = callback

        // Prefer hardware linear acceleration (already gravity-free)
        val sensorToRegister = linearAccelerometer ?: rawAccelerometer
        sensorToRegister?.let { sensor ->
            sensorManager.registerListener(
                this,
                sensor,
                AndroidSensorManager.SENSOR_DELAY_GAME
            )
            isAccelerometerActive = true
        }
    }

    fun stopAccelerometer() {
        linearAccelerometer?.let { sensorManager.unregisterListener(this, it) }
        rawAccelerometer?.let { sensorManager.unregisterListener(this, it) }
        isAccelerometerActive = false
        onAccelerometerChanged = null
    }

    // =========================================================
    // GYROSCOPE
    // =========================================================

    fun startGyroscope(callback: (GyroscopeData) -> Unit) {
        onGyroscopeChanged = callback
        gyroscope?.let { sensor ->
            sensorManager.registerListener(
                this,
                sensor,
                AndroidSensorManager.SENSOR_DELAY_GAME
            )
            isGyroscopeActive = true
        }
    }

    fun stopGyroscope() {
        gyroscope?.let { sensorManager.unregisterListener(this, it) }
        isGyroscopeActive = false
        onGyroscopeChanged = null
    }

    // =========================================================
    // MAGNETOMETER
    // =========================================================

    fun startMagnetometer(callback: (MagnetometerData) -> Unit) {
        onMagnetometerChanged = callback
        magnetometer?.let { sensor ->
            sensorManager.registerListener(
                this,
                sensor,
                AndroidSensorManager.SENSOR_DELAY_GAME
            )
            isMagnetometerActive = true
        }
    }

    fun stopMagnetometer() {
        magnetometer?.let { sensorManager.unregisterListener(this, it) }
        isMagnetometerActive = false
        onMagnetometerChanged = null
        magnetometerAccuracy = AndroidSensorManager.SENSOR_STATUS_UNRELIABLE
    }

    // =========================================================
    // ROTATION VECTOR
    // =========================================================

    fun startRotationVector(callback: (RotationVectorData) -> Unit) {
        onRotationVectorChanged = callback
        rotationVectorSensor?.let { sensor ->
            sensorManager.registerListener(
                this,
                sensor,
                AndroidSensorManager.SENSOR_DELAY_GAME
            )
            isRotationVectorActive = true
        }
    }

    fun stopRotationVector() {
        rotationVectorSensor?.let { sensorManager.unregisterListener(this, it) }
        isRotationVectorActive = false
        onRotationVectorChanged = null
        rotationVectorAccuracy = AndroidSensorManager.SENSOR_STATUS_UNRELIABLE
    }

    // =========================================================
    // SENSOR EVENT LISTENER
    // =========================================================

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        val timestamp = event.timestamp

        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                val data = AccelerometerData(
                    x = event.values[0],
                    y = event.values[1],
                    z = event.values[2],
                    timestampNs = timestamp,
                    isLinearAcceleration = true
                )
                onAccelerometerChanged?.invoke(data)
            }

            Sensor.TYPE_ACCELEROMETER -> {
                // If hardware linear acceleration is active, ignore raw accelerometer
                if (linearAccelerometer != null) return

                // Software high-pass gravity removal
                if (!hasGravity) {
                    gravityValues[0] = event.values[0]
                    gravityValues[1] = event.values[1]
                    gravityValues[2] = event.values[2]
                    hasGravity = true
                } else {
                    gravityValues[0] = gravityAlpha * gravityValues[0] + (1 - gravityAlpha) * event.values[0]
                    gravityValues[1] = gravityAlpha * gravityValues[1] + (1 - gravityAlpha) * event.values[1]
                    gravityValues[2] = gravityAlpha * gravityValues[2] + (1 - gravityAlpha) * event.values[2]
                }

                val linearX = event.values[0] - gravityValues[0]
                val linearY = event.values[1] - gravityValues[1]
                val linearZ = event.values[2] - gravityValues[2]

                val data = AccelerometerData(
                    x = linearX,
                    y = linearY,
                    z = linearZ,
                    timestampNs = timestamp,
                    isLinearAcceleration = false
                )
                onAccelerometerChanged?.invoke(data)
            }

            Sensor.TYPE_GYROSCOPE -> {
                val data = GyroscopeData(
                    x = event.values[0],
                    y = event.values[1],
                    z = event.values[2],
                    timestampNs = timestamp
                )
                onGyroscopeChanged?.invoke(data)
            }

            Sensor.TYPE_MAGNETIC_FIELD -> {
                val data = MagnetometerData(
                    x = event.values[0],
                    y = event.values[1],
                    z = event.values[2],
                    timestampNs = timestamp
                )
                onMagnetometerChanged?.invoke(data)
            }

            Sensor.TYPE_ROTATION_VECTOR -> {
                val rotMatrix = FloatArray(9)
                AndroidSensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
                val data = RotationVectorData(
                    values = event.values.clone(),
                    rotationMatrix = rotMatrix,
                    timestampNs = timestamp
                )
                onRotationVectorChanged?.invoke(data)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        when (sensor?.type) {
            Sensor.TYPE_MAGNETIC_FIELD -> magnetometerAccuracy = accuracy
            Sensor.TYPE_ROTATION_VECTOR -> rotationVectorAccuracy = accuracy
        }
    }
}