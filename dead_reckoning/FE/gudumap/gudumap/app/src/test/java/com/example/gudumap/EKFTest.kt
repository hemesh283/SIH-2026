package com.example.gudumap

import com.example.gudumap.navigation.EKF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EKFTest {

    @Test
    fun testInitialization() {
        val ekf = EKF()
        ekf.initialize(pNorth = 10.0, pEast = 20.0, pDown = 0.0, vNorth = 1.0, vEast = 2.0, vDown = 0.0)

        assertTrue(ekf.isInitialized)
        assertEquals(10.0, ekf.state[0], 1e-6)
        assertEquals(20.0, ekf.state[1], 1e-6)
        assertEquals(0.0, ekf.state[2], 1e-6)
        assertEquals(1.0, ekf.state[3], 1e-6)
        assertEquals(2.0, ekf.state[4], 1e-6)
    }

    @Test
    fun testPredictionStep() {
        val ekf = EKF()
        ekf.initialize(pNorth = 0.0, pEast = 0.0, pDown = 0.0)

        val initialP0 = ekf.P[0][0]

        // Predict displacement of 5m North, 0m East over 1.0s
        ekf.predict(deltaPNed = doubleArrayOf(5.0, 0.0, 0.0), dt = 1.0)

        assertEquals(5.0, ekf.state[0], 1e-6)
        assertEquals(0.0, ekf.state[1], 1e-6)
        assertEquals(5.0, ekf.state[3], 1e-6) // v_N = 5m / 1s = 5m/s

        // Process noise increases covariance during prediction
        assertTrue("Prediction must increase position covariance", ekf.P[0][0] > initialP0)
    }

    @Test
    fun testGnssPositionMeasurementCorrection() {
        val ekf = EKF()
        ekf.initialize(pNorth = 0.0, pEast = 0.0, pDown = 0.0)

        // Predict 10m North
        ekf.predict(deltaPNed = doubleArrayOf(10.0, 0.0, 0.0), dt = 1.0)
        val pBeforeUpdate = ekf.P[0][0]

        // GNSS reports actual position is 12m North with 1m accuracy
        ekf.updateGnssPosition(pNorth = 12.0, pEast = 0.0, pDown = 0.0, noiseMeters = 1.0)

        // State should be pulled towards 12m
        assertTrue("State should be corrected towards GNSS", ekf.state[0] > 10.0 && ekf.state[0] <= 12.0)
        // Covariance should decrease after measurement update
        assertTrue("Measurement update must reduce position uncertainty", ekf.P[0][0] < pBeforeUpdate)
    }

    @Test
    fun testGnssVelocityMeasurementCorrection() {
        val ekf = EKF()
        ekf.initialize(pNorth = 0.0, pEast = 0.0, pDown = 0.0, vNorth = 0.0, vEast = 0.0)

        val vBeforeP = ekf.P[3][3]

        // GNSS velocity measurement of 4.0 m/s North
        ekf.updateGnssVelocity(vNorth = 4.0, vEast = 0.0, vDown = 0.0, noiseMps = 0.2)

        assertTrue("Velocity should update towards GNSS velocity", ekf.state[3] > 2.5 && ekf.state[3] <= 4.0)
        assertTrue("Velocity covariance should decrease", ekf.P[3][3] < vBeforeP)
    }
}
