package com.example.gudumap.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import kotlin.math.sqrt

// =========================================================
// ACCELEROMETER DATA
// =========================================================

data class AccelerometerData(
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f
) {
    val magnitude: Float
        get() = sqrt(
            x * x +
                    y * y +
                    z * z
        )
}

// =========================================================
// GYROSCOPE DATA
// =========================================================

data class GyroscopeData(
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f
) {
    val magnitude: Float
        get() = sqrt(
            x * x +
                    y * y +
                    z * z
        )
}

// =========================================================
// MAGNETOMETER DATA
// =========================================================

data class MagnetometerData(
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f
) {
    val magnitude: Float
        get() = sqrt(
            x * x +
                    y * y +
                    z * z
        )
}

// =========================================================
// SENSOR MANAGER
// =========================================================

class SensorManager(
    context: Context
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE)
                as android.hardware.SensorManager

    // ---------------------------------------------------------
    // SENSORS
    // ---------------------------------------------------------

    private val accelerometer =
        sensorManager.getDefaultSensor(
            Sensor.TYPE_ACCELEROMETER
        )

    private val gyroscope =
        sensorManager.getDefaultSensor(
            Sensor.TYPE_GYROSCOPE
        )

    private val magnetometer =
        sensorManager.getDefaultSensor(
            Sensor.TYPE_MAGNETIC_FIELD
        )

    // ---------------------------------------------------------
    // CALLBACKS
    // ---------------------------------------------------------

    private var onAccelerometerChanged:
            ((AccelerometerData) -> Unit)? = null

    private var onGyroscopeChanged:
            ((GyroscopeData) -> Unit)? = null

    private var onMagnetometerChanged:
            ((MagnetometerData) -> Unit)? = null

    // =========================================================
    // ACCELEROMETER
    // =========================================================

    fun startAccelerometer(
        callback: (AccelerometerData) -> Unit
    ) {

        onAccelerometerChanged = callback

        accelerometer?.let { sensor ->

            sensorManager.registerListener(
                this,
                sensor,
                android.hardware.SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    fun stopAccelerometer() {

        accelerometer?.let { sensor ->

            sensorManager.unregisterListener(
                this,
                sensor
            )
        }

        onAccelerometerChanged = null
    }

    // =========================================================
    // GYROSCOPE
    // =========================================================

    fun startGyroscope(
        callback: (GyroscopeData) -> Unit
    ) {

        onGyroscopeChanged = callback

        gyroscope?.let { sensor ->

            sensorManager.registerListener(
                this,
                sensor,
                android.hardware.SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    fun stopGyroscope() {

        gyroscope?.let { sensor ->

            sensorManager.unregisterListener(
                this,
                sensor
            )
        }

        onGyroscopeChanged = null
    }

    // =========================================================
    // MAGNETOMETER
    // =========================================================

    fun startMagnetometer(
        callback: (MagnetometerData) -> Unit
    ) {

        onMagnetometerChanged = callback

        magnetometer?.let { sensor ->

            sensorManager.registerListener(
                this,
                sensor,
                android.hardware.SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    fun stopMagnetometer() {

        magnetometer?.let { sensor ->

            sensorManager.unregisterListener(
                this,
                sensor
            )
        }

        onMagnetometerChanged = null
    }

    // =========================================================
    // SENSOR EVENTS
    // =========================================================

    override fun onSensorChanged(
        event: SensorEvent?
    ) {

        if (event == null) return

        when (event.sensor.type) {

            // -------------------------------------------------
            // ACCELEROMETER
            // -------------------------------------------------

            Sensor.TYPE_ACCELEROMETER -> {

                val data =
                    AccelerometerData(
                        x = event.values[0],
                        y = event.values[1],
                        z = event.values[2]
                    )

                onAccelerometerChanged
                    ?.invoke(data)
            }

            // -------------------------------------------------
            // GYROSCOPE
            // -------------------------------------------------

            Sensor.TYPE_GYROSCOPE -> {

                val data =
                    GyroscopeData(
                        x = event.values[0],
                        y = event.values[1],
                        z = event.values[2]
                    )

                onGyroscopeChanged
                    ?.invoke(data)
            }

            // -------------------------------------------------
            // MAGNETOMETER
            // -------------------------------------------------

            Sensor.TYPE_MAGNETIC_FIELD -> {

                val data =
                    MagnetometerData(
                        x = event.values[0],
                        y = event.values[1],
                        z = event.values[2]
                    )

                onMagnetometerChanged
                    ?.invoke(data)
            }
        }
    }

    // =========================================================
    // SENSOR ACCURACY
    // =========================================================

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int
    ) {
        // Not needed yet
    }
}