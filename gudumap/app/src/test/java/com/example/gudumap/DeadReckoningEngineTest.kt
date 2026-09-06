package com.example.gudumap

import com.example.gudumap.navigation.DeadReckoningEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin

class DeadReckoningEngineTest {

    @Test
    fun testDeadReckoningDisplacement() {
        val dr = DeadReckoningEngine()
        val initial = dr.getState()
        // 0.0/0.0 is the deliberate "no real fix yet" sentinel (Fix 2, PROJECT_STATUS.md §11)
        // before initialize() has ever been called -- previously hardcoded to Coimbatore.
        assertEquals(0.0, initial.latitude, 0.0001)
        assertEquals(0.0, initial.longitude, 0.0001)

        // Update with ML displacement (10 meters East, 20 meters North)
        val updated = dr.updateWithMLDisplacement(
            globalEastMeters = 10f,
            globalNorthMeters = 20f,
            dtSeconds = 0.5f,
            headingDeg = 45f,
            isStationary = false
        )

        assertTrue(updated.distanceTravelled > 0.0)
        assertTrue(updated.latitude > initial.latitude) // Moved North
        assertTrue(updated.longitude > initial.longitude) // Moved East
        assertFalse(updated.latitude.isNaN())
        assertFalse(updated.longitude.isNaN())
    }

    @Test
    fun testStationaryHandTremorAccumulatesZeroDistance() {
        val dr = DeadReckoningEngine()
        dr.initialize(11.0168, 76.9558)

        // Simulate 60 seconds (6,000 steps at 100 Hz) of hand tremors while standing still
        for (i in 0 until 6000) {
            val tNs = (i + 1) * 10_000_000L
            val phase = (i * 0.04 * 2 * Math.PI).toFloat()
            val accelNorth = 0.22f * sin(phase)
            val accelEast = 0.18f * sin(phase + 1f)

            dr.update(
                accelNorth = accelNorth,
                accelEast = accelEast,
                headingDeg = 0f,
                timestampNs = tNs,
                isStationary = true
            )
        }

        val state = dr.getState()
        assertEquals(0.0, state.distanceTravelled, 1e-4)
        assertEquals(0f, state.speed, 1e-4f)
        assertEquals(11.0168, state.latitude, 1e-6)
        assertEquals(76.9558, state.longitude, 1e-6)
    }

    @Test
    fun testStationaryMLGatingPreventsDrift() {
        val dr = DeadReckoningEngine()
        dr.initialize(11.0168, 76.9558)

        // Simulate ML giving minor noisy output (0.15m) while phone is stationary
        for (i in 0 until 10) {
            dr.updateWithMLDisplacement(
                globalEastMeters = 0.15f,
                globalNorthMeters = 0.10f,
                dtSeconds = 0.5f,
                headingDeg = 0f,
                isStationary = true
            )
        }

        val state = dr.getState()
        assertEquals(0.0, state.distanceTravelled, 1e-4)
        assertEquals(0f, state.speed, 1e-4f)
        assertEquals(11.0168, state.latitude, 1e-6)
        assertEquals(76.9558, state.longitude, 1e-6)
    }

    @Test
    fun testRealMotionAccumulatesDistance() {
        val dr = DeadReckoningEngine()
        dr.initialize(11.0168, 76.9558)

        // Simulate 5 seconds (500 steps) of genuine forward movement
        // Accel pulse of 1.0 m/s^2 forward North for 1 second, then coast
        for (i in 0 until 500) {
            val tNs = (i + 1) * 10_000_000L
            val aN = if (i < 100) 1.0f else 0.0f
            dr.update(
                accelNorth = aN,
                accelEast = 0f,
                headingDeg = 0f,
                timestampNs = tNs,
                isStationary = false
            )
        }

        val state = dr.getState()
        assertTrue("Distance must accumulate during movement", state.distanceTravelled > 0.5)
        assertTrue("Latitude must increase moving North", state.latitude > 11.0168)
    }
}
