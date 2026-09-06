package com.example.gudumap.ui.components

import android.content.Context
import android.graphics.Color as AndroidColor
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.gudumap.R
import com.example.gudumap.map.MapMatcher
import com.example.gudumap.map.OfflineMapManager
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView as OsmMapView
import org.osmdroid.views.overlay.FolderOverlay
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polygon
import org.osmdroid.views.overlay.Polyline

private const val MAX_TRAIL_POINTS = 2000

@Composable
fun MapView(
    latitude: Double,
    longitude: Double,
    headingDeg: Float = 0f,
    mapStatus: String = "OFFLINE",
    offlineMapStatus: String = "AVAILABLE",
    roadName: String = "",
    blackoutMode: Boolean = false,
    naiveLatitude: Double = latitude,
    naiveLongitude: Double = longitude,
    uncertaintyRadiusMeters: Double = 0.0,
    modifier: Modifier = Modifier
) {
    // Trails and blackout-transition tracking live at composable scope so they survive
    // recomposition but reset cleanly on a fresh blackout (mirrors DeadReckoningEngine
    // resetting NaiveIntegrator on setBlackoutMode(true)).
    val correctedTrail = remember { mutableListOf<GeoPoint>() }
    val naiveTrail = remember { mutableListOf<GeoPoint>() }
    val wasBlackout = remember { mutableStateOf(false) }
    AndroidView(
        modifier = modifier
            .fillMaxWidth()
            .height(320.dp),
        factory = { context: Context ->
            val offlineManager = OfflineMapManager(context)
            val tileProvider = offlineManager.createOfflineTileProvider()

            val mapView = if (tileProvider != null) {
                OsmMapView(context, tileProvider)
            } else {
                OsmMapView(context)
            }

            mapView.apply {
                setMultiTouchControls(true)
                // STRICT OFFLINE: Disable all network data connections
                setUseDataConnection(false)
                minZoomLevel = 11.0
                maxZoomLevel = 18.0
                controller.setZoom(15.5)

                val initialPoint = if (latitude > 1.0 && longitude > 1.0) {
                    GeoPoint(latitude, longitude)
                } else {
                    OfflineMapManager.COIMBATORE_CENTER
                }
                controller.setCenter(initialPoint)

                // Optional: add real road vector overlays for visual clarity
                try {
                    val matcher = MapMatcher(context)
                    val roads = matcher.getRoads()
                    if (roads.isNotEmpty()) {
                        val roadsOverlay = FolderOverlay()
                        roadsOverlay.name = "coimbatore_vector_roads"
                        for (road in roads) {
                            val polyline = Polyline(this)
                            polyline.outlinePaint.color = AndroidColor.argb(90, 59, 130, 246)
                            polyline.outlinePaint.strokeWidth = 3f
                            val pts = road.points.map { GeoPoint(it.lat, it.lon) }
                            polyline.setPoints(pts)
                            polyline.title = road.name
                            roadsOverlay.add(polyline)
                        }
                        overlays.add(roadsOverlay)
                    }
                } catch (e: Exception) {
                    // Fallback to raster tiles only
                }

                // Vehicle Position Marker
                val marker = Marker(this).apply {
                    id = "vehicle_marker"
                    position = initialPoint
                    title = if (roadName.isNotBlank()) roadName else "Gudumap Position"
                    snippet = "Lat: %.5f, Lon: %.5f".format(initialPoint.latitude, initialPoint.longitude)
                    rotation = headingDeg
                    setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
                    val iconDrawable = ContextCompat.getDrawable(context, R.drawable.ic_navigation_arrow)
                    if (iconDrawable != null) {
                        icon = iconDrawable
                    }
                }
                overlays.add(marker)

                invalidate()
            }
        },
        update = { map ->
            // Enforce offline mode on every recomposition
            map.setUseDataConnection(false)

            val hasRealFix = latitude > 1.0 && longitude > 1.0
            val currentPoint = if (hasRealFix) {
                GeoPoint(latitude, longitude)
            } else {
                // No real fix yet -- fall back to a generic map viewport only (Coimbatore is the
                // only offline tileset bundled), never claim this as an actual measured position.
                OfflineMapManager.COIMBATORE_CENTER
            }

            // A fresh blackout starting resets both trails, mirroring the engine
            // resetting its own naive/corrected integrators at the same moment.
            if (blackoutMode && !wasBlackout.value) {
                correctedTrail.clear()
                naiveTrail.clear()
            }
            wasBlackout.value = blackoutMode

            correctedTrail.add(currentPoint)
            if (correctedTrail.size > MAX_TRAIL_POINTS) correctedTrail.removeAt(0)

            if (blackoutMode) {
                val naivePoint = if (naiveLatitude > 1.0 && naiveLongitude > 1.0) {
                    GeoPoint(naiveLatitude, naiveLongitude)
                } else {
                    currentPoint
                }
                naiveTrail.add(naivePoint)
                if (naiveTrail.size > MAX_TRAIL_POINTS) naiveTrail.removeAt(0)
            }

            val existingMarker = map.overlays.filterIsInstance<Marker>().firstOrNull { it.id == "vehicle_marker" }
            if (hasRealFix) {
                val marker = existingMarker
                    ?: Marker(map).also {
                        it.id = "vehicle_marker"
                        it.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
                        val iconDrawable = ContextCompat.getDrawable(map.context, R.drawable.ic_navigation_arrow)
                        if (iconDrawable != null) {
                            it.icon = iconDrawable
                        }
                        map.overlays.add(it)
                    }

                marker.position = currentPoint
                marker.rotation = headingDeg
                marker.title = if (roadName.isNotBlank()) roadName else "Gudumap Position"
                marker.snippet = "Lat: %.5f, Lon: %.5f".format(currentPoint.latitude, currentPoint.longitude)
            } else if (existingMarker != null) {
                // No real fix (yet, or anymore) -- don't show a marker that could be mistaken
                // for an actual position (Fix 2, PROJECT_STATUS.md §11).
                map.overlays.remove(existingMarker)
            }

            // Corrected (GRU+EKF+ZUPT) trail -- blue.
            val correctedPolyline = map.overlays
                .filterIsInstance<Polyline>()
                .firstOrNull { it.id == "corrected_trail" }
                ?: Polyline(map).also {
                    it.id = "corrected_trail"
                    it.outlinePaint.color = AndroidColor.rgb(37, 99, 235)
                    it.outlinePaint.strokeWidth = 7f
                    map.overlays.add(0, it) // beneath the road overlay/marker
                }
            correctedPolyline.setPoints(correctedTrail)

            // Naive (uncorrected double-integration) trail -- red, only exists once a
            // blackout has started, so the contrast only appears when it's meaningful.
            val naivePolyline = map.overlays
                .filterIsInstance<Polyline>()
                .firstOrNull { it.id == "naive_trail" }
                ?: Polyline(map).also {
                    it.id = "naive_trail"
                    it.outlinePaint.color = AndroidColor.argb(200, 220, 38, 38)
                    it.outlinePaint.strokeWidth = 5f
                    map.overlays.add(0, it)
                }
            naivePolyline.setPoints(if (blackoutMode) naiveTrail else emptyList())

            // Growing EKF position-uncertainty circle around the current corrected
            // position -- only meaningful (and only shown) during an active blackout.
            val uncertaintyCircle = map.overlays
                .filterIsInstance<Polygon>()
                .firstOrNull { it.id == "uncertainty_circle" }
                ?: Polygon(map).also {
                    it.id = "uncertainty_circle"
                    it.fillPaint.color = AndroidColor.argb(40, 220, 38, 38)
                    it.outlinePaint.color = AndroidColor.argb(120, 220, 38, 38)
                    it.outlinePaint.strokeWidth = 2f
                    map.overlays.add(0, it)
                }
            if (blackoutMode && uncertaintyRadiusMeters > 0.5) {
                uncertaintyCircle.setPoints(Polygon.pointsAsCircle(currentPoint, uncertaintyRadiusMeters))
            } else {
                uncertaintyCircle.setPoints(emptyList())
            }

            map.controller.animateTo(currentPoint)
            map.invalidate()
        }
    )
}