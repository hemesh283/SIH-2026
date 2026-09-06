package com.example.gudumap.navigation

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Classical coordinate frame transformation utilities for dead reckoning.
 *
 * Handles:
 * 1. Phone-to-Vehicle frame transformation (R_p2v)
 * 2. Window local displacement to world NED frame (R_start * delta_p_local)
 * 3. Local metric displacement (North, East) to geodetic WGS-84 (Lat, Lon)
 */
class CoordinateTransformer {

    companion object {
        const val WGS84_A = 6378137.0         // Semi-major axis (meters)
        const val WGS84_E2 = 0.00669437999014  // First eccentricity squared
        const val MEAN_EARTH_RADIUS = 6371000.0 // Mean radius (meters)

        /**
         * Calculates great-circle distance between two geodetic points in meters.
         * Pure Kotlin implementation (zero dependency on android.location.Location.distanceBetween).
         */
        fun computeDistanceBetween(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val dLat = Math.toRadians(lat2 - lat1)
            val dLon = Math.toRadians(lon2 - lon1)
            val a = kotlin.math.sin(dLat / 2) * kotlin.math.sin(dLat / 2) +
                    kotlin.math.cos(Math.toRadians(lat1)) * kotlin.math.cos(Math.toRadians(lat2)) *
                    kotlin.math.sin(dLon / 2) * kotlin.math.sin(dLon / 2)
            val c = 2 * kotlin.math.atan2(kotlin.math.sqrt(a), kotlin.math.sqrt(1 - a))
            return MEAN_EARTH_RADIUS * c
        }
    }

    // Phone-to-vehicle rotation matrix (3x3 row-major)
    // Defaults to identity (phone aligned with vehicle frame)
    private var phoneToVehicleMatrix: FloatArray = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    )

    /**
     * Sets mounting angles for phone-to-vehicle transformation.
     *
     * @param rollDeg Roll angle around X axis in degrees
     * @param pitchDeg Pitch angle around Y axis in degrees
     * @param yawDeg Yaw angle around Z axis in degrees
     */
    fun setMountAngles(rollDeg: Float, pitchDeg: Float, yawDeg: Float) {
        val roll = Math.toRadians(rollDeg.toDouble()).toFloat()
        val pitch = Math.toRadians(pitchDeg.toDouble()).toFloat()
        val yaw = Math.toRadians(yawDeg.toDouble()).toFloat()

        val cr = cos(roll)
        val sr = sin(roll)
        val cp = cos(pitch)
        val sp = sin(pitch)
        val cy = cos(yaw)
        val sy = sin(yaw)

        // R = Rz(yaw) * Ry(pitch) * Rx(roll)
        phoneToVehicleMatrix = floatArrayOf(
            cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr
        )
    }

    /**
     * Transform a 3D vector from phone frame to vehicle frame.
     */
    fun transformPhoneToVehicle(phoneVec: FloatArray): FloatArray {
        require(phoneVec.size == 3) { "Vector must have 3 elements" }
        val out = FloatArray(3)
        out[0] = phoneToVehicleMatrix[0] * phoneVec[0] + phoneToVehicleMatrix[1] * phoneVec[1] + phoneToVehicleMatrix[2] * phoneVec[2]
        out[1] = phoneToVehicleMatrix[3] * phoneVec[0] + phoneToVehicleMatrix[4] * phoneVec[1] + phoneToVehicleMatrix[5] * phoneVec[2]
        out[2] = phoneToVehicleMatrix[6] * phoneVec[0] + phoneToVehicleMatrix[7] * phoneVec[1] + phoneToVehicleMatrix[8] * phoneVec[2]
        return out
    }

    /**
     * Rotate local displacement predicted by GRU into world navigation frame (NED: North-East-Down).
     *
     * Contract:
     * Model outputs delta_p_local = [dx, dy, dz] in initial window local frame.
     * delta_p_ned = R_start * delta_p_local
     *
     * @param localDisplacement [dx, dy, dz] in meters
     * @param headingDeg Heading in degrees (clockwise from North: 0 = North, 90 = East)
     * @return FloatArray of 3 elements [delta_North, delta_East, delta_Down] in meters
     */
    fun rotateLocalToWorld(
        localDisplacement: FloatArray,
        headingDeg: Float
    ): FloatArray {
        val headingRad = Math.toRadians(headingDeg.toDouble())
        val cosH = cos(headingRad).toFloat()
        val sinH = sin(headingRad).toFloat()

        val dx = localDisplacement[0]
        val dy = localDisplacement[1]
        val dz = localDisplacement[2]

        val deltaNorth = (cosH * dx - sinH * dy)
        val deltaEast = (sinH * dx + cosH * dy)
        val deltaDown = dz

        return floatArrayOf(deltaNorth, deltaEast, deltaDown)
    }

    /**
     * Rotate local displacement using full 3x3 rotation matrix R_start.
     */
    fun rotateLocalToWorldWithMatrix(
        localDisplacement: FloatArray,
        rotationMatrix3x3: FloatArray
    ): FloatArray {
        require(localDisplacement.size == 3) { "localDisplacement must have 3 elements" }
        require(rotationMatrix3x3.size == 9) { "rotationMatrix3x3 must have 9 elements" }

        val out = FloatArray(3)
        out[0] = rotationMatrix3x3[0] * localDisplacement[0] + rotationMatrix3x3[1] * localDisplacement[1] + rotationMatrix3x3[2] * localDisplacement[2]
        out[1] = rotationMatrix3x3[3] * localDisplacement[0] + rotationMatrix3x3[4] * localDisplacement[1] + rotationMatrix3x3[5] * localDisplacement[2]
        out[2] = rotationMatrix3x3[6] * localDisplacement[0] + rotationMatrix3x3[7] * localDisplacement[1] + rotationMatrix3x3[8] * localDisplacement[2]
        return out
    }

    /**
     * Converts metric displacements [deltaNorth, deltaEast] to change in [latitude, longitude].
     * Uses WGS-84 ellipsoidal curvature radii.
     *
     * @param latDeg Current latitude in degrees
     * @param deltaNorth North displacement in meters
     * @param deltaEast East displacement in meters
     * @return DoubleArray of 2 elements [newLatDeg, newLonDeg]
     */
    fun addMetricDisplacementToGeodetic(
        latDeg: Double,
        lonDeg: Double,
        deltaNorth: Double,
        deltaEast: Double
    ): DoubleArray {
        val latRad = Math.toRadians(latDeg)
        val sinLat = sin(latRad)

        // Meridian radius of curvature
        val denom = sqrt(1.0 - WGS84_E2 * sinLat * sinLat)
        val rm = WGS84_A * (1.0 - WGS84_E2) / (denom * denom * denom)
        // Prime vertical radius of curvature
        val rn = WGS84_A / denom

        val deltaLatRad = deltaNorth / rm
        val deltaLonRad = if (abs(cos(latRad)) > 1e-6) {
            deltaEast / (rn * cos(latRad))
        } else {
            0.0
        }

        val newLat = latDeg + Math.toDegrees(deltaLatRad)
        val newLon = lonDeg + Math.toDegrees(deltaLonRad)
        return doubleArrayOf(newLat, newLon)
    }
}
