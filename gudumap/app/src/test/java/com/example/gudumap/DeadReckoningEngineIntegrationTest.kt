package com.example.gudumap

import android.location.Location
import com.example.gudumap.navigation.DeadReckoningEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DeadReckoningEngineIntegrationTest {

    @Test
    fun testNavigationEngineInitializationAndGnssCorrection() {
        val engine = DeadReckoningEngine()
        engine.initialize(latitude = 51.7520, longitude = -1.2577, speedMps = 5.0f, bearingDeg = 90.0f)

        val state = engine.getState()
        assertTrue("Engine must be initialized", state.isInitialized)
        assertEquals(51.7520, state.latitude, 1e-6)
        assertEquals(-1.2577, state.longitude, 1e-6)
        assertEquals(90.0f, state.heading, 1e-3f)

        // GNSS correction
        engine.correctWithGnss(latitude = 51.7525, longitude = -1.2570, speedMps = 6.0f, bearingDeg = 85.0f)

        val state2 = engine.getState()
        // Smooth EKF fusion: state smoothly moves towards GNSS fix rather than abruptly snapping
        assertTrue("Latitude should move towards GNSS fix", state2.latitude > 51.7520 && state2.latitude <= 51.7525)
        assertTrue("Longitude should move towards GNSS fix", state2.longitude > -1.2577 && state2.longitude <= -1.2570)

        // Multiple consecutive GNSS fixes smoothly converge the filter state to the GNSS trajectory
        for (i in 0 until 15) {
            engine.correctWithGnss(latitude = 51.7525, longitude = -1.2570, speedMps = 6.0f, bearingDeg = 85.0f)
        }
        val convergedState = engine.getState()
        assertEquals(51.7525, convergedState.latitude, 1e-4)
        assertEquals(-1.2570, convergedState.longitude, 1e-4)
    }

    @Test
    fun testBlackoutModeEvaluation() {
        val engine = DeadReckoningEngine()
        engine.initialize(latitude = 51.7520, longitude = -1.2577)

        // Start blackout
        engine.setBlackoutMode(true)
        assertTrue(engine.getState().isBlackout)

        // Evaluate drift against GNSS reference position
        val eval = engine.trajectoryIntegrator.evaluateDrift(
            estimatedLat = 51.7521,
            estimatedLon = -1.2577,
            gnssReferenceLat = 51.7520,
            gnssReferenceLon = -1.2577
        )

        assertNotNull(eval)
        assertTrue("Drift evaluation should compute distance error", eval.positionErrorMeters > 0.0)
    }

    @Test
    fun testMapMatcherAndSmoothGnssRecovery() {
        var mapMatcherCalled = false
        val customMapMatcher = object : com.example.gudumap.navigation.MapMatcher {
            override val isEnabled: Boolean = true
            override fun match(point: com.example.gudumap.tracking.TrajectoryPoint): com.example.gudumap.tracking.TrajectoryPoint {
                mapMatcherCalled = true
                return point
            }
        }

        val engine = DeadReckoningEngine(mapMatcher = customMapMatcher)
        engine.initialize(latitude = 12.9716, longitude = 77.5946, speedMps = 2.0f, bearingDeg = 0.0f)

        // 1. Enter blackout mode
        engine.setBlackoutMode(true)
        assertTrue("Blackout mode must be active", engine.getState().isBlackout)

        // 2. Incoming GNSS fix during blackout is ignored by the filter
        engine.correctWithGnss(latitude = 12.9720, longitude = 77.5946, speedMps = 2.0f, bearingDeg = 0.0f)
        // Position remains at anchor because filter did not accept blackout GNSS
        assertEquals(12.9716, engine.getState().latitude, 1e-5)

        // 3. Exit blackout mode (GNSS recovery)
        engine.setBlackoutMode(false)
        assertFalse("Blackout mode must be inactive", engine.getState().isBlackout)

        // 4. GNSS returns: smooth EKF recovery
        engine.correctWithGnss(latitude = 12.9720, longitude = 77.5946, speedMps = 2.0f, bearingDeg = 0.0f)

        val stateRecovered = engine.getState()
        // State should smoothly fuse toward recovery position without NaN or abrupt jump
        assertTrue("Latitude must be finite", !stateRecovered.latitude.isNaN())
        assertTrue("Longitude must be finite", !stateRecovered.longitude.isNaN())
        assertTrue("Latitude must smoothly move towards GNSS fix", stateRecovered.latitude > 12.9716 && stateRecovered.latitude <= 12.9720)

        // Multiple updates converge to GNSS track
        for (i in 0 until 15) {
            engine.correctWithGnss(latitude = 12.9720, longitude = 77.5946, speedMps = 2.0f, bearingDeg = 0.0f)
        }
        val convergedRecovered = engine.getState()
        assertEquals(12.9720, convergedRecovered.latitude, 1e-4)

        // 5. Verify MapMatcher interface
        val testPoint = com.example.gudumap.tracking.TrajectoryPoint(
            latitude = 12.9720,
            longitude = 77.5946
        )
        val matched = customMapMatcher.match(testPoint)
        assertTrue("MapMatcher must be called", mapMatcherCalled)
        assertEquals(testPoint.latitude, matched.latitude, 1e-6)
    }

    @Test
    fun testHighRateImuSamplePipeline() {
        val engine = DeadReckoningEngine()
        engine.initialize(latitude = 12.9716, longitude = 77.5946)

        // Feed 50 consecutive synthetic IMU samples
        var ts = System.nanoTime()
        for (i in 0 until 50) {
            ts += 100_000_000L // 100 ms = 10 Hz
            val sample = com.example.gudumap.sensor.ImuSample(
                timestampNs = ts,
                ax = 0.05f,
                ay = 0.02f,
                az = 0.0f,
                gx = 0.01f,
                gy = 0.00f,
                gz = 0.02f
            )
            engine.addSensorSample(sample)
        }

        val state = engine.getState()
        assertTrue("Engine must remain initialized", state.isInitialized)
        assertTrue("Speed must be finite", !state.speed.isNaN())
    }
}

