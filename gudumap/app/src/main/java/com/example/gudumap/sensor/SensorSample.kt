package com.example.gudumap.sensor

/**
 * Data structures for sensor measurements.
 */

/**
 * Synchronized IMU sample.
 *
 * @param timestampNs Sensor timestamp in nanoseconds
 * @param ax Linear acceleration X in m/s^2 (gravity-removed)
 * @param ay Linear acceleration Y in m/s^2 (gravity-removed)
 * @param az Linear acceleration Z in m/s^2 (gravity-removed)
 * @param gx Angular velocity X in rad/s
 * @param gy Angular velocity Y in rad/s
 * @param gz Angular velocity Z in rad/s
 */
data class ImuSample(
    val timestampNs: Long,
    val ax: Float,
    val ay: Float,
    val az: Float,
    val gx: Float,
    val gy: Float,
    val gz: Float
) {
    /**
     * Converts to model feature array [acc_x(g), acc_y(g), acc_z(g), gyro_x(rad/s), gyro_y(rad/s), gyro_z(rad/s)].
     */
    fun toFeatureArray(gravityMps2: Float = 9.80665f): FloatArray {
        return floatArrayOf(
            ax / gravityMps2,
            ay / gravityMps2,
            az / gravityMps2,
            gx,
            gy,
            gz
        )
    }
}

/**
 * Orientation sample derived from TYPE_ROTATION_VECTOR.
 *
 * @param timestampNs Timestamp in nanoseconds
 * @param rotationMatrix 3x3 rotation matrix (9 floats)
 * @param quaternion Unit quaternion [w, x, y, z] or [x, y, z, w]
 * @param azimuthRad Azimuth/Yaw in radians
 * @param pitchRad Pitch in radians
 * @param rollRad Roll in radians
 */
data class OrientationSample(
    val timestampNs: Long,
    val rotationMatrix: FloatArray,
    val quaternion: FloatArray,
    val azimuthRad: Float,
    val pitchRad: Float,
    val rollRad: Float
) {
    val headingDegrees: Float
        get() {
            var deg = Math.toDegrees(azimuthRad.toDouble()).toFloat()
            if (deg < 0f) deg += 360f
            return deg
        }
}
