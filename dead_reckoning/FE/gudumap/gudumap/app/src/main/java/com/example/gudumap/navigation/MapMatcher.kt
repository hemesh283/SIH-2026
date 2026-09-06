package com.example.gudumap.navigation

import com.example.gudumap.tracking.TrajectoryPoint

/**
 * Interface for optional map-matching post-processors.
 *
 * In accordance with the project contract:
 * - Map matching must remain distinct from raw navigation filtering and visualization.
 * - Nearest-road snapping must not be used to artificially conceal navigation drift.
 * - For MVP, PassThroughMapMatcher preserves the true authoritative dead-reckoning trajectory.
 */
interface MapMatcher {
    val isEnabled: Boolean

    /**
     * Map-match a navigation trajectory point to road network geometry.
     *
     * @param point Authoritative fused navigation trajectory point
     * @return Matched trajectory point (or identical point if passthrough)
     */
    fun match(point: TrajectoryPoint): TrajectoryPoint
}

/**
 * Default pass-through implementation ensuring navigation errors and drift
 * are accurately visible on the map rather than artificially obscured.
 */
class PassThroughMapMatcher : MapMatcher {
    override val isEnabled: Boolean = false

    override fun match(point: TrajectoryPoint): TrajectoryPoint {
        return point
    }
}
