package com.example.gudumap

import com.example.gudumap.navigation.CoordinateTransformer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CoordinateTransformerTest {

    private val transformer = CoordinateTransformer()

    @Test
    fun testDefaultPhoneToVehicleIsIdentity() {
        val phoneVec = floatArrayOf(1.5f, -2.5f, 3.5f)
        val vehVec = transformer.transformPhoneToVehicle(phoneVec)

        assertEquals(phoneVec[0], vehVec[0], 1e-5f)
        assertEquals(phoneVec[1], vehVec[1], 1e-5f)
        assertEquals(phoneVec[2], vehVec[2], 1e-5f)
    }

    @Test
    fun testMountAnglesRotation() {
        // 90 degree yaw rotation: phone X (right) -> vehicle Y, phone Y (up) -> vehicle -X
        val customTransformer = CoordinateTransformer()
        customTransformer.setMountAngles(rollDeg = 0f, pitchDeg = 0f, yawDeg = 90f)

        val phoneForward = floatArrayOf(1.0f, 0.0f, 0.0f)
        val rotated = customTransformer.transformPhoneToVehicle(phoneForward)

        assertEquals(0.0f, rotated[0], 1e-5f)
        assertEquals(1.0f, rotated[1], 1e-5f)
        assertEquals(0.0f, rotated[2], 1e-5f)
    }

    @Test
    fun testRotateLocalToWorldHeadingOrientations() {
        val localDisplacement = floatArrayOf(10.0f, 0.0f, 0.0f) // 10m forward

        // 1. Heading = 0 deg (North)
        val worldNorth = transformer.rotateLocalToWorld(localDisplacement, headingDeg = 0f)
        assertEquals(10.0f, worldNorth[0], 1e-4f) // North = 10m
        assertEquals(0.0f, worldNorth[1], 1e-4f)  // East = 0m
        assertEquals(0.0f, worldNorth[2], 1e-4f)  // Down = 0m

        // 2. Heading = 90 deg (East)
        val worldEast = transformer.rotateLocalToWorld(localDisplacement, headingDeg = 90f)
        assertEquals(0.0f, worldEast[0], 1e-4f)   // North = 0m
        assertEquals(10.0f, worldEast[1], 1e-4f)  // East = 10m
        assertEquals(0.0f, worldEast[2], 1e-4f)   // Down = 0m

        // 3. Heading = 180 deg (South)
        val worldSouth = transformer.rotateLocalToWorld(localDisplacement, headingDeg = 180f)
        assertEquals(-10.0f, worldSouth[0], 1e-4f) // North = -10m
        assertEquals(0.0f, worldSouth[1], 1e-4f)   // East = 0m

        // 4. Heading = 270 deg (West)
        val worldWest = transformer.rotateLocalToWorld(localDisplacement, headingDeg = 270f)
        assertEquals(0.0f, worldWest[0], 1e-4f)    // North = 0m
        assertEquals(-10.0f, worldWest[1], 1e-4f)  // East = -10m
    }

    @Test
    fun testGeodeticWgs84Displacement() {
        val originLat = 51.7520 // Oxford latitude
        val originLon = -1.2577

        // Move 111,111 meters North (~1 degree of latitude)
        val result = transformer.addMetricDisplacementToGeodetic(
            latDeg = originLat,
            lonDeg = originLon,
            deltaNorth = 111111.0,
            deltaEast = 0.0
        )

        val newLat = result[0]
        val newLon = result[1]

        assertTrue("New latitude should increase", newLat > originLat)
        assertEquals(originLat + 1.0, newLat, 0.02)
        assertEquals(originLon, newLon, 1e-6)
    }
}
