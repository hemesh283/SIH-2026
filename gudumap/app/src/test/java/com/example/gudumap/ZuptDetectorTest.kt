package com.example.gudumap

import com.example.gudumap.navigation.ZuptDetector
import com.example.gudumap.sensor.ImuSample
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ZuptDetectorTest {

    @Test
    fun testDetectsStationaryWithMultiSignalCombination() {
        val detector = ZuptDetector(
            minConsecutiveSamples = 5
        )

        // Feed quiet sensor samples (low accel magnitude & variance, low gyro magnitude & variance)
        for (i in 0 until 15) {
            val sample = ImuSample(
                timestampNs = i * 10_000_000L,
                ax = 0.01f,
                ay = 0.01f,
                az = 0.01f,
                gx = 0.005f,
                gy = 0.005f,
                gz = 0.005f
            )
            detector.update(sample, gnssSpeed = 0.0f)
        }

        assertTrue("Multi-signal quiet sensor should trigger stationary state", detector.isStationary)
    }

    @Test
    fun testRejectsStationaryWhenGyroscopeIsActive() {
        val detector = ZuptDetector(
            minConsecutiveSamples = 5,
            gyroMagnitudeThreshold = 0.10f
        )

        // Accel is low, but phone is rotating (gyro is high)
        for (i in 0 until 15) {
            val sample = ImuSample(
                timestampNs = i * 10_000_000L,
                ax = 0.01f,
                ay = 0.01f,
                az = 0.01f,
                gx = 0.50f, // High angular velocity (e.g. rotating in hand or turning)
                gy = 0.0f,
                gz = 0.0f
            )
            detector.update(sample, gnssSpeed = 0.0f)
        }

        assertFalse("Active rotation must prevent stationary detection", detector.isStationary)
    }

    @Test
    fun testRejectsStationaryWhenGnssSpeedIsHigh() {
        val detector = ZuptDetector(
            minConsecutiveSamples = 5,
            gnssSpeedThreshold = 0.30f
        )

        // Sensors quiet, but vehicle is cruising at 15 m/s
        for (i in 0 until 15) {
            val sample = ImuSample(
                timestampNs = i * 10_000_000L,
                ax = 0.01f,
                ay = 0.01f,
                az = 0.01f,
                gx = 0.01f,
                gy = 0.01f,
                gz = 0.01f
            )
            detector.update(sample, gnssSpeed = 15.0f)
        }

        assertFalse("High GNSS speed must prevent stationary detection", detector.isStationary)
    }

    @Test
    fun testEnableDisableModularity() {
        val detector = ZuptDetector(
            minConsecutiveSamples = 3
        )

        // Feed quiet samples
        for (i in 0 until 10) {
            detector.update(
                ImuSample(i * 10_000_000L, 0f, 0f, 0f, 0f, 0f, 0f),
                gnssSpeed = 0f
            )
        }
        assertTrue(detector.isStationary)

        // Disable detector
        detector.isEnabled = false
        assertFalse("Disabled detector must report non-stationary", detector.isStationary)
    }
}
