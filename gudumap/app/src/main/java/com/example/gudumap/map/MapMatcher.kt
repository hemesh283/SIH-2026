package com.example.gudumap.map

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.InputStreamReader
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class RoadPoint(val lat: Double, val lon: Double)

data class RoadSegment(
    val id: Long,
    val name: String,
    val type: String,
    val bearing: Float,
    val points: List<RoadPoint>
)

data class MapMatchResult(
    val matchedLat: Double,
    val matchedLon: Double,
    val roadName: String,
    val confidence: Float,
    val distanceToRoadMeters: Float,
    val isSnapped: Boolean
)

class MapMatcher(context: Context? = null) {

    companion object {
        private const val TAG = "Gudumap:MapMatcher"
        private const val EARTH_RADIUS = 6_371_000.0
        private const val MAX_DISTANCE_THRESHOLD_METERS = 40.0
        private const val MAX_HEADING_DIFF_DEG = 50.0
        private const val MIN_CONFIDENCE_SNAP = 0.35f
    }

    private val roads = ArrayList<RoadSegment>()

    init {
        context?.let { loadRoadsFromAssets(it) }
    }

    fun loadRoadsFromAssets(context: Context) {
        try {
            val stream = try {
                context.assets.open("maps/coimbatore/coimbatore_roads.json")
            } catch (e: Exception) {
                context.assets.open("maps/coimbatore_roads.json")
            }
            val jsonStr = InputStreamReader(stream).use { it.readText() }
            val json = JSONObject(jsonStr)
            val roadsArray = json.getJSONArray("roads")

            roads.clear()
            for (i in 0 until roadsArray.length()) {
                val roadObj = roadsArray.getJSONObject(i)
                val id = roadObj.optLong("id", i.toLong())
                val name = roadObj.optString("name", "Coimbatore Road")
                val type = roadObj.optString("type", "primary")
                val bearing = roadObj.optDouble("bearing", 0.0).toFloat()

                val pointsArray = roadObj.getJSONArray("points")
                val points = ArrayList<RoadPoint>(pointsArray.length())
                for (j in 0 until pointsArray.length()) {
                    val p = pointsArray.getJSONObject(j)
                    points.add(RoadPoint(p.getDouble("lat"), p.getDouble("lon")))
                }

                if (points.size >= 2) {
                    roads.add(RoadSegment(id, name, type, bearing, points))
                }
            }
            Log.i(TAG, "Loaded ${roads.size} Coimbatore road segments for map matching.")
        } catch (e: Exception) {
            Log.w(TAG, "Could not load coimbatore_roads.json: ${e.message}")
        }
    }

    /**
     * Matches the estimated position against Coimbatore roads.
     * Uses distance and heading alignment to compute confidence.
     * If confidence is low, keeps the original estimated position.
     */
    fun match(
        estimatedLat: Double,
        estimatedLon: Double,
        headingDeg: Float,
        speedKmh: Float
    ): MapMatchResult {
        if (roads.isEmpty()) {
            return MapMatchResult(
                matchedLat = estimatedLat,
                matchedLon = estimatedLon,
                roadName = "",
                confidence = 0f,
                distanceToRoadMeters = 0f,
                isSnapped = false
            )
        }

        var bestDistance = Double.MAX_VALUE
        var bestSnappedLat = estimatedLat
        var bestSnappedLon = estimatedLon
        var bestRoadName = ""
        var bestConfidence = 0f

        for (road in roads) {
            val points = road.points
            for (i in 0 until points.size - 1) {
                val p1 = points[i]
                val p2 = points[i + 1]

                // Fast spatial bounding box filter (~200m latitude/longitude threshold) --
                // merged back from teammate's branch, PROJECT_STATUS.md §26. Cheaply skips
                // segments nowhere near the estimated position before the more expensive
                // projection/distance math below -- addresses the brute-force-scan performance
                // concern flagged in §16 (no spatial index, linear scan over every road segment).
                if (abs(p1.lat - estimatedLat) > 0.002 && abs(p2.lat - estimatedLat) > 0.002) {
                    continue
                }
                if (abs(p1.lon - estimatedLon) > 0.002 && abs(p2.lon - estimatedLon) > 0.002) {
                    continue
                }

                val proj = projectPointOntoSegment(estimatedLat, estimatedLon, p1.lat, p1.lon, p2.lat, p2.lon)
                val dist = distanceBetweenMeters(estimatedLat, estimatedLon, proj.lat, proj.lon)

                if (dist < MAX_DISTANCE_THRESHOLD_METERS && dist < bestDistance) {
                    // Calculate road segment bearing
                    val segmentBearing = calculateBearing(p1.lat, p1.lon, p2.lat, p2.lon)

                    // Angular difference (accounting for bidirectional roads)
                    var headingDiff = abs(headingDeg - segmentBearing) % 360f
                    if (headingDiff > 180f) headingDiff = 360f - headingDiff

                    var reverseDiff = abs(headingDeg - ((segmentBearing + 180f) % 360f)) % 360f
                    if (reverseDiff > 180f) reverseDiff = 360f - reverseDiff

                    val minHeadingDiff = minOf(headingDiff, reverseDiff)

                    if (minHeadingDiff <= MAX_HEADING_DIFF_DEG || speedKmh < 3f) {
                        val distScore = (1.0 - (dist / MAX_DISTANCE_THRESHOLD_METERS)).coerceIn(0.0, 1.0).toFloat()
                        val angleScore = (1.0 - (minHeadingDiff / MAX_HEADING_DIFF_DEG)).coerceIn(0.0, 1.0).toFloat()
                        val confidence = distScore * 0.6f + angleScore * 0.4f

                        if (confidence > bestConfidence) {
                            bestDistance = dist
                            bestSnappedLat = proj.lat
                            bestSnappedLon = proj.lon
                            bestRoadName = road.name
                            bestConfidence = confidence
                        }
                    }
                }
            }
        }

        val shouldSnap = bestConfidence >= MIN_CONFIDENCE_SNAP && bestDistance < MAX_DISTANCE_THRESHOLD_METERS

        return if (shouldSnap) {
            MapMatchResult(
                matchedLat = bestSnappedLat,
                matchedLon = bestSnappedLon,
                roadName = bestRoadName,
                confidence = bestConfidence,
                distanceToRoadMeters = bestDistance.toFloat(),
                isSnapped = true
            )
        } else {
            MapMatchResult(
                matchedLat = estimatedLat,
                matchedLon = estimatedLon,
                roadName = if (bestDistance < 100.0) bestRoadName else "",
                confidence = 0f,
                distanceToRoadMeters = if (bestDistance < 1000.0) bestDistance.toFloat() else 0f,
                isSnapped = false
            )
        }
    }

    private fun projectPointOntoSegment(
        lat: Double, lon: Double,
        lat1: Double, lon1: Double,
        lat2: Double, lon2: Double
    ): RoadPoint {
        val latRad = Math.toRadians(lat)
        val lonScale = cos(latRad)

        // Local flat projection in meters
        val x = Math.toRadians(lon) * lonScale * EARTH_RADIUS
        val y = Math.toRadians(lat) * EARTH_RADIUS

        val x1 = Math.toRadians(lon1) * lonScale * EARTH_RADIUS
        val y1 = Math.toRadians(lat1) * EARTH_RADIUS

        val x2 = Math.toRadians(lon2) * lonScale * EARTH_RADIUS
        val y2 = Math.toRadians(lat2) * EARTH_RADIUS

        val dx = x2 - x1
        val dy = y2 - y1
        val len2 = dx * dx + dy * dy

        if (len2 == 0.0) return RoadPoint(lat1, lon1)

        val t = ((x - x1) * dx + (y - y1) * dy) / len2
        val clampedT = t.coerceIn(0.0, 1.0)

        val projX = x1 + clampedT * dx
        val projY = y1 + clampedT * dy

        val projLat = Math.toDegrees(projY / EARTH_RADIUS)
        val projLon = Math.toDegrees(projX / (EARTH_RADIUS * lonScale))

        return RoadPoint(projLat, projLon)
    }

    private fun distanceBetweenMeters(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2) * sin(dLat / 2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
                sin(dLon / 2) * sin(dLon / 2)
        val c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return EARTH_RADIUS * c
    }

    private fun calculateBearing(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Float {
        val phi1 = Math.toRadians(lat1)
        val phi2 = Math.toRadians(lat2)
        val deltaLambda = Math.toRadians(lon2 - lon1)

        val y = sin(deltaLambda) * cos(phi2)
        val x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(deltaLambda)
        val bearing = Math.toDegrees(atan2(y, x)).toFloat()
        return (bearing + 360f) % 360f
    }

    fun getRoads(): List<RoadSegment> = roads
}
