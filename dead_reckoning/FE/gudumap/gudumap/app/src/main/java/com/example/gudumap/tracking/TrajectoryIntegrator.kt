package com.example.gudumap.tracking

import android.location.Location
import kotlin.math.max

data class TrajectoryPoint(
    val latitude: Double,
    val longitude: Double,
    val altitude: Double = 0.0,
    val speedMps: Float = 0f,
    val headingDeg: Float = 0f,
    val timestampNs: Long = System.nanoTime(),
    val isBlackout: Boolean = false,
    val isStationary: Boolean = false
)

data class BlackoutEvaluation(
    val positionErrorMeters: Double = 0.0,
    val driftPercentage: Double = 0.0,
    val blackoutDistanceMeters: Double = 0.0
)

/**
 * Manages trajectory history, distance calculation, and blackout evaluation metrics.
 */
class TrajectoryIntegrator(
    private val maxPoints: Int = 1000
) {

    private val lock = Any()
    private val points = ArrayList<TrajectoryPoint>(maxPoints)

    private var totalDistanceTravelled: Double = 0.0
    private var blackoutStartDistance: Double = 0.0
    private var blackoutStartPoint: TrajectoryPoint? = null

    /**
     * Append a new estimated navigation point.
     */
    fun addPoint(point: TrajectoryPoint) {
        synchronized(lock) {
            if (points.isNotEmpty()) {
                val last = points.last()
                val dist = com.example.gudumap.navigation.CoordinateTransformer.computeDistanceBetween(
                    last.latitude, last.longitude,
                    point.latitude, point.longitude
                )
                totalDistanceTravelled += dist
            }

            points.add(point)
            if (points.size > maxPoints) {
                // Remove oldest point
                points.removeAt(0)
            }
        }
    }

    /**
     * Mark the start of a GNSS blackout evaluation period.
     */
    fun startBlackout(currentPoint: TrajectoryPoint) {
        synchronized(lock) {
            blackoutStartPoint = currentPoint
            blackoutStartDistance = totalDistanceTravelled
        }
    }

    /**
     * Evaluate drift during blackout against reference GNSS location.
     */
    fun evaluateDrift(
        estimatedLat: Double,
        estimatedLon: Double,
        gnssReferenceLat: Double,
        gnssReferenceLon: Double
    ): BlackoutEvaluation {
        val posError = com.example.gudumap.navigation.CoordinateTransformer.computeDistanceBetween(
            estimatedLat, estimatedLon,
            gnssReferenceLat, gnssReferenceLon
        )
        val blackoutDistance = max(0.0, totalDistanceTravelled - blackoutStartDistance)
        val driftPct = if (blackoutDistance > 1.0) {
            (posError / blackoutDistance) * 100.0
        } else {
            0.0
        }

        return BlackoutEvaluation(
            positionErrorMeters = posError,
            driftPercentage = driftPct,
            blackoutDistanceMeters = blackoutDistance
        )
    }

    fun getTotalDistance(): Double = synchronized(lock) { totalDistanceTravelled }

    fun getPoints(): List<TrajectoryPoint> = synchronized(lock) { ArrayList(points) }

    fun reset() {
        synchronized(lock) {
            points.clear()
            totalDistanceTravelled = 0.0
            blackoutStartDistance = 0.0
            blackoutStartPoint = null
        }
    }
}
