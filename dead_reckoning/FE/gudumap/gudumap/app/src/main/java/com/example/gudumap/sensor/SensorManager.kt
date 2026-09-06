package com.example.gudumap.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager as AndroidSensorManager

/**
 * SensorManager providing high-rate IMU streams for dead reckoning.
 *
 * Preferred sources:
 * - Linear Acceleration: Sensor.TYPE_LINEAR_ACCELERATION (gravity-removed, m/s^2)
 * - Gyroscope: Sensor.TYPE_GYROSCOPE (rad/s)
 * - Orientation: Sensor.TYPE_ROTATION_VECTOR
 */
class SensorManager(
    context: Context,
    private val samplingPeriodUs: Int = AndroidSensorManager.SENSOR_DELAY_FASTEST
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as AndroidSensorManager

    private val linearAccelSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val gyroSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private val rotationVectorSensor: Sensor? =
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    // Callbacks
    var onImuSampleListener: ((ImuSample) -> Unit)? = null
    var onOrientationListener: ((OrientationSample) -> Unit)? = null

    // Cache latest values for timestamp synchronization
    private val lock = Any()
    private var lastLinearAccel = FloatArray(3)
    private var lastLinearAccelTimestampNs: Long = 0L
    private var hasLinearAccel = false

    private var lastGyro = FloatArray(3)
    private var lastGyroTimestampNs: Long = 0L
    private var hasGyro = false

    private val tempRotationMatrix = FloatArray(9)
    private val tempOrientation = FloatArray(3)

    private var isRunning = false

    fun start() {
        synchronized(lock) {
            if (isRunning) return
            isRunning = true

            linearAccelSensor?.let {
                sensorManager.registerListener(this, it, samplingPeriodUs)
            }
            gyroSensor?.let {
                sensorManager.registerListener(this, it, samplingPeriodUs)
            }
            rotationVectorSensor?.let {
                sensorManager.registerListener(this, it, samplingPeriodUs)
            }
        }
    }

    fun stop() {
        synchronized(lock) {
            if (!isRunning) return
            isRunning = false
            sensorManager.unregisterListener(this)
            hasLinearAccel = false
            hasGyro = false
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION, Sensor.TYPE_ACCELEROMETER -> {
                var dispatchSample: ImuSample? = null
                synchronized(lock) {
                    lastLinearAccel[0] = event.values[0]
                    lastLinearAccel[1] = event.values[1]
                    lastLinearAccel[2] = event.values[2]
                    lastLinearAccelTimestampNs = event.timestamp
                    hasLinearAccel = true

                    if (hasGyro) {
                        dispatchSample = ImuSample(
                            timestampNs = event.timestamp,
                            ax = lastLinearAccel[0],
                            ay = lastLinearAccel[1],
                            az = lastLinearAccel[2],
                            gx = lastGyro[0],
                            gy = lastGyro[1],
                            gz = lastGyro[2]
                        )
                    }
                }
                dispatchSample?.let { onImuSampleListener?.invoke(it) }
            }

            Sensor.TYPE_GYROSCOPE -> {
                var dispatchSample: ImuSample? = null
                synchronized(lock) {
                    lastGyro[0] = event.values[0]
                    lastGyro[1] = event.values[1]
                    lastGyro[2] = event.values[2]
                    lastGyroTimestampNs = event.timestamp
                    hasGyro = true

                    if (hasLinearAccel) {
                        dispatchSample = ImuSample(
                            timestampNs = event.timestamp,
                            ax = lastLinearAccel[0],
                            ay = lastLinearAccel[1],
                            az = lastLinearAccel[2],
                            gx = lastGyro[0],
                            gy = lastGyro[1],
                            gz = lastGyro[2]
                        )
                    }
                }
                dispatchSample?.let { onImuSampleListener?.invoke(it) }
            }

            Sensor.TYPE_ROTATION_VECTOR -> {
                val rotMatrix = FloatArray(9)
                AndroidSensorManager.getRotationMatrixFromVector(rotMatrix, event.values)

                val quat = FloatArray(4)
                AndroidSensorManager.getQuaternionFromVector(quat, event.values)

                AndroidSensorManager.getOrientation(rotMatrix, tempOrientation)

                val sample = OrientationSample(
                    timestampNs = event.timestamp,
                    rotationMatrix = rotMatrix,
                    quaternion = quat,
                    azimuthRad = tempOrientation[0],
                    pitchRad = tempOrientation[1],
                    rollRad = tempOrientation[2]
                )
                onOrientationListener?.invoke(sample)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
