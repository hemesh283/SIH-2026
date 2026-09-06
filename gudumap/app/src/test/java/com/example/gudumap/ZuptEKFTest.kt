package com.example.gudumap

import com.example.gudumap.navigation.EKF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ZuptEKFTest {

    @Test
    fun testZuptAsEkfPseudoMeasurementDoesNotHardClamp() {
        val ekf = EKF(zuptVelNoise = 0.05)
        // Initialize with initial moving velocity of 2.0 m/s North
        ekf.initialize(pNorth = 100.0, pEast = 50.0, pDown = 0.0, vNorth = 2.0, vEast = 0.0, vDown = 0.0)

        val vBefore = ekf.state[3]
        val pBefore = ekf.state[0]
        val velCovBefore = ekf.P[3][3]

        // Apply single ZUPT pseudo-measurement update
        ekf.updateZupt()

        val vAfter = ekf.state[3]
        val velCovAfter = ekf.P[3][3]

        // 1. Velocity must decrease towards zero
        assertTrue("Velocity should decrease towards zero after ZUPT", vAfter < vBefore)
        assertTrue("Velocity should still be positive (not hard-clamped immediately to 0)", vAfter > 0.0)

        // 2. Velocity covariance must decrease (more certain that velocity is near zero)
        assertTrue("Velocity covariance should decrease after ZUPT update", velCovAfter < velCovBefore)

        // 3. Position must not jump abruptly
        assertEquals("Position should remain stable", pBefore, ekf.state[0], 0.1)
    }

    @Test
    fun testSequentialZuptConvergesVelocitySmoothly() {
        val ekf = EKF(zuptVelNoise = 0.02)
        ekf.initialize(pNorth = 0.0, pEast = 0.0, pDown = 0.0, vNorth = 3.0, vEast = 2.0, vDown = 0.0)

        // Apply several ZUPT pseudo-measurements
        for (i in 0 until 10) {
            ekf.updateZupt()
        }

        // Velocity should smoothly converge close to zero (< 0.05 m/s)
        assertTrue("North velocity should converge near zero", abs(ekf.state[3]) < 0.05)
        assertTrue("East velocity should converge near zero", abs(ekf.state[4]) < 0.05)
    }
}
