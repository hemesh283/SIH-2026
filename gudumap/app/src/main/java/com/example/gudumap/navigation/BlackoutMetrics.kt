package com.example.gudumap.navigation

/**
 * Explicit metrics tracked during and after a GNSS blackout event.
 *
 * Tracks authoritative DR performance against reference GNSS ground-truth
 * without allowing GNSS to modify or corrupt the navigation filter.
 */
data class BlackoutMetrics(
    val blackoutStartTime: Long = 0L,
    val blackoutEndTime: Long = 0L,
    val blackoutDurationSeconds: Double = 0.0,
    val blackoutStartLatitude: Double = 0.0,
    val blackoutStartLongitude: Double = 0.0,
    val blackoutEndGnssLatitude: Double = 0.0,
    val blackoutEndGnssLongitude: Double = 0.0,
    val drLatitude: Double = 0.0,
    val drLongitude: Double = 0.0,
    val gnssReferenceDistance: Double = 0.0,
    val drDistance: Double = 0.0,
    val positionErrorMeters: Double = 0.0,
    val distanceErrorMeters: Double = 0.0,
    val driftPercentage: Double = 0.0,
    val maximumPositionErrorMeters: Double = 0.0,
    val maximumSpeedKmh: Float = 0f,
    val averageSpeedKmh: Float = 0f,
    val mlInferenceMs: Long = 0L,
    val numberOfAcceptedMLPredictions: Int = 0,
    val numberOfClampedMLPredictions: Int = 0,
    val numberOfRejectedMLPredictions: Int = 0,
    val stationaryDuration: Double = 0.0,
    val rotatingInPlaceDuration: Double = 0.0
)
