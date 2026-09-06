package com.example.gudumap

import com.example.gudumap.navigation.EKF
import com.example.gudumap.navigation.NHC
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class NHCTest {

    @Test
    fun testNhcRestrictsLateralAndVerticalVelocity() {
        val ekf = EKF()
        // Heading North (0 deg), vehicle moving North at 10 m/s with unwanted lateral East slip of 3 m/s and vertical 2 m/s
        ekf.initialize(pNorth = 0.0, pEast = 0.0, pDown = 0.0, vNorth = 10.0, vEast = 3.0, vDown = 2.0)

        val nhc = NHC(isEnabled = true, lateralNoiseMps = 0.1, verticalNoiseMps = 0.1)
        nhc.applyConstraint(ekf, headingDeg = 0f)

        // Forward velocity North should be largely preserved (~10 m/s)
        assertTrue("Forward velocity should remain high", ekf.state[3] > 9.0)

        // Lateral slip East should be significantly reduced
        assertTrue("Lateral velocity should be reduced by NHC", ekf.state[4] < 3.0)

        // Vertical velocity Down should be significantly reduced
        assertTrue("Vertical velocity should be reduced by NHC", ekf.state[5] < 2.0)
    }

    @Test
    fun testNhcDisabledDoesNotModifyState() {
        val ekf = EKF()
        ekf.initialize(pNorth = 0.0, pEast = 0.0, pDown = 0.0, vNorth = 5.0, vEast = 2.0, vDown = 1.0)

        val nhc = NHC(isEnabled = false)
        nhc.applyConstraint(ekf, headingDeg = 0f)

        // Values must remain exactly identical
        assertEquals(5.0, ekf.state[3], 1e-6)
        assertEquals(2.0, ekf.state[4], 1e-6)
        assertEquals(1.0, ekf.state[5], 1e-6)
    }
}
