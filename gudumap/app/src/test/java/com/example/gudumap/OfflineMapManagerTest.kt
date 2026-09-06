package com.example.gudumap

import com.example.gudumap.map.OfflineMapManager
import com.example.gudumap.map.OfflineMapStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OfflineMapManagerTest {

    @Test
    fun testCoimbatoreCoordinatesAndConstants() {
        assertEquals(11.0168, OfflineMapManager.COIMBATORE_DEFAULT_LAT, 1e-4)
        assertEquals(76.9558, OfflineMapManager.COIMBATORE_DEFAULT_LON, 1e-4)
        assertEquals(11, OfflineMapManager.MIN_ZOOM)
        assertEquals(17, OfflineMapManager.MAX_ZOOM)

        assertNotNull(OfflineMapManager.COIMBATORE_BOUNDS)
        assertTrue(OfflineMapManager.COIMBATORE_BOUNDS.latNorth > OfflineMapManager.COIMBATORE_BOUNDS.latSouth)
        assertTrue(OfflineMapManager.COIMBATORE_BOUNDS.lonEast > OfflineMapManager.COIMBATORE_BOUNDS.lonWest)

        // Verify bounding box covers central Coimbatore
        assertTrue(OfflineMapManager.COIMBATORE_DEFAULT_LAT < OfflineMapManager.COIMBATORE_BOUNDS.latNorth)
        assertTrue(OfflineMapManager.COIMBATORE_DEFAULT_LAT > OfflineMapManager.COIMBATORE_BOUNDS.latSouth)
        assertTrue(OfflineMapManager.COIMBATORE_DEFAULT_LON < OfflineMapManager.COIMBATORE_BOUNDS.lonEast)
        assertTrue(OfflineMapManager.COIMBATORE_DEFAULT_LON > OfflineMapManager.COIMBATORE_BOUNDS.lonWest)
    }

    @Test
    fun testOfflineMapStatusEnum() {
        val statuses = OfflineMapStatus.values()
        assertTrue(statuses.contains(OfflineMapStatus.AVAILABLE))
        assertTrue(statuses.contains(OfflineMapStatus.LOADING))
        assertTrue(statuses.contains(OfflineMapStatus.ERROR))
        assertTrue(statuses.contains(OfflineMapStatus.NOT_AVAILABLE))
    }
}
