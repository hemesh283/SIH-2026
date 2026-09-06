package com.example.gudumap.ui.components

import android.content.Context
import android.preference.PreferenceManager

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView

import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.OnlineTileSourceBase
import org.osmdroid.util.GeoPoint
import org.osmdroid.util.MapTileIndex
import org.osmdroid.views.MapView as OsmMapView
import org.osmdroid.views.overlay.Marker

import android.location.Location

@Composable
fun MapView(
    location: Location?,
    modifier: Modifier = Modifier
) {

    AndroidView(
        modifier = modifier
            .fillMaxWidth()
            .height(300.dp),

        factory = { context: Context ->

            Configuration.getInstance().load(
                context,
                PreferenceManager.getDefaultSharedPreferences(context)
            )

            Configuration.getInstance().userAgentValue =
                "Gudumap/1.0 (SIH26168; Android)"

            OsmMapView(context).apply {

                // -----------------------------------------------------
                // Map source
                // -----------------------------------------------------

                val esriStreetMap = object : OnlineTileSourceBase(
                    "EsriWorldStreetMap",
                    0,
                    19,
                    256,
                    ".png",
                    arrayOf(
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/"
                    )
                ) {

                    override fun getTileURLString(
                        pMapTileIndex: Long
                    ): String {

                        val zoom =
                            MapTileIndex.getZoom(pMapTileIndex)

                        val x =
                            MapTileIndex.getX(pMapTileIndex)

                        val y =
                            MapTileIndex.getY(pMapTileIndex)

                        return "${baseUrl}$zoom/$y/$x.png"
                    }
                }

                setTileSource(esriStreetMap)

                setMultiTouchControls(true)

                setBuiltInZoomControls(true)

                controller.setZoom(17.0)

                // -----------------------------------------------------
                // Initial position
                // -----------------------------------------------------

                val initialPoint = if (location != null) {

                    GeoPoint(
                        location.latitude,
                        location.longitude
                    )

                } else {

                    GeoPoint(
                        12.9716,
                        77.5946
                    )
                }

                controller.setCenter(initialPoint)

                // -----------------------------------------------------
                // Vehicle marker
                // -----------------------------------------------------

                val marker = Marker(this)

                marker.position = initialPoint

                marker.title = "You are here"

                marker.snippet =
                    "Gudumap current position"

                overlays.add(marker)

                invalidate()
            }
        },

        update = { map ->

            if (location != null) {

                val currentPoint = GeoPoint(
                    location.latitude,
                    location.longitude
                )

                // Find our marker
                val marker =
                    map.overlays
                        .filterIsInstance<Marker>()
                        .firstOrNull()

                if (marker != null) {

                    marker.position = currentPoint

                } else {

                    val newMarker = Marker(map)

                    newMarker.position = currentPoint

                    newMarker.title = "You are here"

                    map.overlays.add(newMarker)
                }

                // Move camera to current location
                map.controller.animateTo(currentPoint)

                map.invalidate()
            }
        }
    )
}