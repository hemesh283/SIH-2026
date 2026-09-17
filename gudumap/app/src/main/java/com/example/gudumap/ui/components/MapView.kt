package com.example.gudumap.ui.components

import android.content.Context
import android.graphics.Color as AndroidColor
import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.gudumap.R
import com.example.gudumap.map.MapMatcher
import com.example.gudumap.map.OfflineMapManager
import org.osmdroid.tileprovider.tilesource.ITileSource
import org.osmdroid.tileprovider.tilesource.XYTileSource
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView as OsmMapView
import org.osmdroid.views.overlay.FolderOverlay
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polygon
import org.osmdroid.views.overlay.Polyline

private const val MAX_TRAIL_POINTS = 2000
private const val TAG = "Gudumap:MapView"

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
    isExpanded: Boolean = false,
    isDarkMode: Boolean = false,
    onToggleExpand: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    val correctedTrail = remember { mutableListOf<GeoPoint>() }
    val naiveTrail = remember { mutableListOf<GeoPoint>() }
    val wasBlackout = remember { mutableStateOf(false) }

    var osmMapRef by remember { mutableStateOf<OsmMapView?>(null) }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .then(if (isExpanded) Modifier.fillMaxSize() else Modifier.height(340.dp))
            .clip(RoundedCornerShape(if (isExpanded) 0.dp else 20.dp))
            .border(
                if (isExpanded) 0.dp else 1.dp,
                if (isDarkMode) Color(0xFF334155) else Color(0xFFE2E8F0),
                RoundedCornerShape(if (isExpanded) 0.dp else 20.dp)
            )
    ) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { context: Context ->
                org.osmdroid.config.Configuration.getInstance().tileFileSystemCacheMaxBytes = 100L * 1024L * 1024L
                org.osmdroid.config.Configuration.getInstance().tileFileSystemCacheTrimBytes = 80L * 1024L * 1024L

                val offlineManager = OfflineMapManager(context)
                val tileProvider = offlineManager.createOfflineTileProvider()

                val mapView = if (tileProvider != null) {
                    Log.i(TAG, "Offline tile provider ready (status=${offlineManager.getOfflineMapStatusString()}, " +
                        "tiles=${offlineManager.getTileCount()}) -- using bundled coimbatore.mbtiles")
                    OsmMapView(context, tileProvider)
                } else {
                    Log.w(TAG, "Offline tile provider unavailable (status=${offlineManager.getOfflineMapStatusString()}) " +
                        "-- rendering with no base tiles; NOT falling back to any online source")
                    val noNetworkSource: ITileSource = XYTileSource(
                        "GudumapNoNetwork",
                        OfflineMapManager.MIN_ZOOM,
                        OfflineMapManager.MAX_ZOOM,
                        256,
                        ".png",
                        emptyArray()
                    )
                    OsmMapView(context).apply {
                        setTileSource(noNetworkSource)
                    }
                }

                mapView.apply {
                    setMultiTouchControls(true)
                    setUseDataConnection(false)
                    minZoomLevel = OfflineMapManager.MIN_ZOOM.toDouble()
                    maxZoomLevel = OfflineMapManager.MAX_ZOOM.toDouble()
                    controller.setZoom(15.5)

                    val initialPoint = if (latitude > 1.0 && longitude > 1.0) {
                        GeoPoint(latitude, longitude)
                    } else {
                        OfflineMapManager.COIMBATORE_CENTER
                    }
                    controller.setCenter(initialPoint)

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

                osmMapRef = mapView
                mapView
            },
            update = { map ->
                map.setUseDataConnection(false)
                val colorFilter = if (isDarkMode) {
                    val matrix = floatArrayOf(
                        -0.85f, 0f, 0f, 0f, 230f,
                        0f, -0.85f, 0f, 0f, 230f,
                        0f, 0f, -0.85f, 0f, 240f,
                        0f, 0f, 0f, 1.0f, 0f
                    )
                    android.graphics.ColorMatrixColorFilter(matrix)
                } else {
                    null
                }
                map.overlayManager.tilesOverlay.setColorFilter(colorFilter)

                val hasRealFix = latitude > 1.0 && longitude > 1.0
                val currentPoint = if (hasRealFix) {
                    GeoPoint(latitude, longitude)
                } else {
                    OfflineMapManager.COIMBATORE_CENTER
                }

                if (blackoutMode && !wasBlackout.value) {
                    correctedTrail.clear()
                    naiveTrail.clear()
                }
                wasBlackout.value = blackoutMode

                if (blackoutMode) {
                    correctedTrail.add(currentPoint)
                    if (correctedTrail.size > MAX_TRAIL_POINTS) correctedTrail.removeAt(0)
                }

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
                    marker.title = if (roadName.isNotBlank()) "📍 $roadName (Your Location)" else "📍 Your Location"
                    marker.snippet = "Lat: %.5f, Lon: %.5f".format(currentPoint.latitude, currentPoint.longitude)
                } else if (existingMarker != null) {
                    map.overlays.remove(existingMarker)
                }

                val correctedPolyline = map.overlays
                    .filterIsInstance<Polyline>()
                    .firstOrNull { it.id == "corrected_trail" }
                    ?: Polyline(map).also {
                        it.id = "corrected_trail"
                        it.outlinePaint.color = AndroidColor.rgb(37, 99, 235)
                        it.outlinePaint.strokeWidth = 7f
                        map.overlays.add(0, it)
                    }
                correctedPolyline.setPoints(if (blackoutMode) correctedTrail else emptyList())

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

                val pinpointRing = map.overlays
                    .filterIsInstance<Polygon>()
                    .firstOrNull { it.id == "pinpoint_ring" }
                    ?: Polygon(map).also {
                        it.id = "pinpoint_ring"
                        it.fillPaint.color = AndroidColor.argb(35, 37, 99, 235)
                        it.outlinePaint.color = AndroidColor.argb(160, 37, 99, 235)
                        it.outlinePaint.strokeWidth = 3f
                        map.overlays.add(0, it)
                    }
                if (hasRealFix) {
                    pinpointRing.setPoints(Polygon.pointsAsCircle(currentPoint, 8.0))
                } else {
                    pinpointRing.setPoints(emptyList())
                }

                map.controller.setCenter(currentPoint)
                map.invalidate()
            }
        )

        // ========================================================
        // FLOATING MAP OVERLAY CONTROLS (UX Upgrade)
        // ========================================================

        // Top-Left: Offline Map Status Badge
        Surface(
            modifier = Modifier
                .align(Alignment.TopStart)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(start = if (isExpanded) 62.dp else 12.dp, top = 12.dp),
            shape = RoundedCornerShape(20.dp),
            color = Color(0xEE0F172A),
            shadowElevation = 4.dp
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(7.dp)
                        .background(Color(0xFF10B981), CircleShape)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "COIMBATORE OFFLINE",
                    fontSize = 10.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = Color.White,
                    letterSpacing = 0.5.sp
                )
            }
        }

        // Top-Right: Heading Compass Pill
        Surface(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(12.dp),
            shape = RoundedCornerShape(20.dp),
            color = Color(0xEE0F172A),
            shadowElevation = 4.dp
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "🧭 ${headingDeg.toInt()}°",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF60A5FA)
                )
            }
        }

        // Bottom-Left: My Location Floating Action Pill
        Surface(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(12.dp)
                .clickable {
                    val currentPoint = if (latitude > 1.0 && longitude > 1.0) {
                        GeoPoint(latitude, longitude)
                    } else {
                        OfflineMapManager.COIMBATORE_CENTER
                    }
                    osmMapRef?.controller?.setZoom(16.0)
                    osmMapRef?.controller?.setCenter(currentPoint)
                    val marker = osmMapRef?.overlays?.filterIsInstance<Marker>()?.firstOrNull { it.id == "vehicle_marker" }
                    marker?.showInfoWindow()
                    osmMapRef?.invalidate()
                },
            shape = RoundedCornerShape(20.dp),
            color = Color(0xFF2563EB),
            shadowElevation = 4.dp
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "📍 MY LOCATION",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = Color.White,
                    letterSpacing = 0.5.sp
                )
            }
        }

        // Bottom-Right: Zoom, Recenter & Enlarge Controls
        Column(
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Enlarge / Fullscreen Map Button
            if (onToggleExpand != null) {
                Surface(
                    modifier = Modifier
                        .size(38.dp)
                        .clickable { onToggleExpand() },
                    shape = CircleShape,
                    color = Color(0xFF0F172A),
                    shadowElevation = 4.dp
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text(if (isExpanded) "↙" else "⛶", fontSize = 16.sp, color = Color.White, fontWeight = FontWeight.Bold)
                    }
                }
            }

            // Recenter Button
            Surface(
                modifier = Modifier
                    .size(38.dp)
                    .clickable {
                        val currentPoint = if (latitude > 1.0 && longitude > 1.0) {
                            GeoPoint(latitude, longitude)
                        } else {
                            OfflineMapManager.COIMBATORE_CENTER
                        }
                        osmMapRef?.controller?.setCenter(currentPoint)
                        osmMapRef?.invalidate()
                    },
                shape = CircleShape,
                color = Color.White,
                shadowElevation = 4.dp
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Text("🎯", fontSize = 16.sp)
                }
            }

            // Zoom In Button
            Surface(
                modifier = Modifier
                    .size(34.dp)
                    .clickable {
                        osmMapRef?.controller?.zoomIn()
                    },
                shape = CircleShape,
                color = Color.White,
                shadowElevation = 4.dp
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Text("+", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1E293B))
                }
            }

            // Zoom Out Button
            Surface(
                modifier = Modifier
                    .size(34.dp)
                    .clickable {
                        osmMapRef?.controller?.zoomOut()
                    },
                shape = CircleShape,
                color = Color.White,
                shadowElevation = 4.dp
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Text("-", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1E293B))
                }
            }
        }
    }
}
