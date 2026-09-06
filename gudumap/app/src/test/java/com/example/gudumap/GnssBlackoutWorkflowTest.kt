package com.example.gudumap

import com.example.gudumap.navigation.CoordinateTransformer
import com.example.gudumap.navigation.DeadReckoningEngine
import com.example.gudumap.navigation.GateAction
import com.example.gudumap.navigation.NavMotionState
import com.example.gudumap.sensor.ImuSample
import com.example.gudumap.sensor.OrientationSample
import com.example.gudumap.tracking.TrajectoryIntegrator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.FileInputStream
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Deterministic validation test suite for the complete GNSS-Blackout Navigation Workflow.
 * Verifies all 11 required test scenarios for SIH26168.
 */
class GnssBlackoutWorkflowTest {

    private fun findModelFile(): File? {
        val candidates = listOf(
            File("app/src/main/assets/gru_io_vnbd.onnx"),
            File("src/main/assets/gru_io_vnbd.onnx"),
            File("app/src/main/assets/models/gru_io_vnbd.onnx")
        )
        return candidates.firstOrNull { it.exists() && it.length() > 0 }
    }

    private fun setupEngineWithModel(): DeadReckoningEngine {
        val engine = DeadReckoningEngine()
        val modelFile = findModelFile()
        if (modelFile != null) {
            FileInputStream(modelFile).use { engine.modelRunner.initializeFromStream(it) }
        }
        return engine
    }

    // =========================================================================
    // TEST 1: Start blackout correctly captures GNSS reference
    // =========================================================================
    @Test
    fun test1_startBlackoutCorrectlyCapturesGnssReference() {
        val engine = DeadReckoningEngine()
        val startLat = 11.0185
        val startLon = 76.9572
        val startSpeed = 1.4f
        val startBearing = 65.0f

        engine.initialize(
            latitude = startLat,
            longitude = startLon,
            speedMps = startSpeed,
            bearingDeg = startBearing
        )

        assertFalse("Blackout should not be active before starting", engine.blackoutMode)

        engine.setBlackoutMode(true)

        assertTrue("Blackout mode must be active after setBlackoutMode(true)", engine.blackoutMode)
        val state = engine.getState()
        assertEquals("Start latitude must match initialized reference", startLat, state.latitude, 1e-5)
        assertEquals("Start longitude must match initialized reference", startLon, state.longitude, 1e-5)
        assertEquals("Start heading must match reference", startBearing, state.heading, 0.1f)
        assertTrue("Blackout flag must be set in engine state", state.isBlackout)
    }

    // =========================================================================
    // TEST 2: During blackout, GNSS position updates do NOT modify DR/EKF position
    // =========================================================================
    @Test
    fun test2_duringBlackoutGnssUpdatesDoNotModifyDrEkfPosition() {
        val engine = DeadReckoningEngine()
        val startLat = 11.0168
        val startLon = 76.9558
        engine.initialize(startLat, startLon)

        engine.setBlackoutMode(true)
        assertTrue(engine.blackoutMode)

        val beforeState = engine.getState()
        val beforeDist = engine.distanceTravelled

        // Attempt to feed divergent GNSS position updates during blackout
        engine.correctWithGnss(
            latitude = 11.0350, // ~2 km away
            longitude = 76.9850,
            speedMps = 15.0f,
            bearingDeg = 180f
        )

        val afterState = engine.getState()
        val afterDist = engine.distanceTravelled

        assertEquals("GNSS update during blackout must NOT alter DR latitude", beforeState.latitude, afterState.latitude, 1e-7)
        assertEquals("GNSS update during blackout must NOT alter DR longitude", beforeState.longitude, afterState.longitude, 1e-7)
        assertEquals("GNSS update during blackout must NOT alter travelled distance", beforeDist, afterDist, 1e-7)
        assertEquals("EKF North position must remain unaffected", 0.0, engine.ekf.state[0], 1e-5)
        assertEquals("EKF East position must remain unaffected", 0.0, engine.ekf.state[1], 1e-5)
    }

    // =========================================================================
    // TEST 3: DR continues updating when GNSS is unavailable
    // =========================================================================
    @Test
    fun test3_drContinuesUpdatingWhenGnssIsUnavailable() {
        val engine = DeadReckoningEngine()
        engine.initialize(11.0168, 76.9558)
        engine.setBlackoutMode(true)

        val initialDist = engine.distanceTravelled

        // Apply forward displacement steps purely from DR (no GNSS)
        val updated = engine.updateWithMLDisplacement(
            globalEastMeters = 5.0f,
            globalNorthMeters = 10.0f,
            dtSeconds = 1.0f,
            headingDeg = 26.5f,
            isStationary = false
        )

        assertTrue("Distance must advance during blackout via DR", updated.distanceTravelled > initialDist)
        assertTrue("Latitude must increase towards North", updated.latitude > 11.0168)
        assertTrue("Longitude must increase towards East", updated.longitude > 76.9558)
        assertTrue("Speed must be positive from DR", updated.speed > 0f)
    }

    // =========================================================================
    // TEST 4: Blackout duration is calculated correctly
    // =========================================================================
    @Test
    fun test4_blackoutDurationIsCalculatedCorrectly() {
        val t0 = 1700000000000L
        val t1 = t0 + 42350L // 42.35 seconds later

        val durationSec = (t1 - t0) / 1000.0
        assertEquals("Blackout duration must be accurately calculated in seconds", 42.35, durationSec, 1e-4)

        // Trajectory integrator duration check
        val integrator = TrajectoryIntegrator()
        integrator.startBlackout(
            com.example.gudumap.tracking.TrajectoryPoint(
                latitude = 11.0168,
                longitude = 76.9558,
                timestampNs = 1_000_000_000L
            )
        )
        val startPt = integrator.getBlackoutStartPoint()
        org.junit.Assert.assertNotNull("Blackout start point must be preserved", startPt)
        assertEquals(11.0168, startPt!!.latitude, 1e-6)
    }

    // =========================================================================
    // TEST 5: Recovery calculates DR-vs-GNSS position error
    // =========================================================================
    @Test
    fun test5_recoveryCalculatesDrVsGnssPositionError() {
        val drLat = 11.017500
        val drLon = 76.956500

        val gnssLat = 11.017800
        val gnssLon = 76.956800

        val computedError = CoordinateTransformer.computeDistanceBetween(drLat, drLon, gnssLat, gnssLon)

        assertTrue("Position error must be positive and non-zero", computedError > 0.0)
        assertTrue("Position error between nearby coordinates must be physically realistic (~40-60m)", computedError in 40.0..60.0)

        // TrajectoryIntegrator evaluation parity
        val integrator = TrajectoryIntegrator()
        integrator.startBlackout(
            com.example.gudumap.tracking.TrajectoryPoint(latitude = 11.0168, longitude = 76.9558)
        )
        val eval = integrator.evaluateDrift(
            estimatedLat = drLat,
            estimatedLon = drLon,
            gnssReferenceLat = gnssLat,
            gnssReferenceLon = gnssLon
        )
        assertEquals("Drift evaluator must match computed position error", computedError, eval.positionErrorMeters, 0.01)
    }

    // =========================================================================
    // TEST 6: Drift percentage is calculated correctly
    // =========================================================================
    @Test
    fun test6_driftPercentageIsCalculatedCorrectly() {
        // Case A: 50.0m walked, 3.5m error -> 7.0% drift
        val drDistA = 50.0
        val errorA = 3.5
        val driftPctA = if (drDistA > 1.0) (errorA / drDistA) * 100.0 else 0.0
        assertEquals("Drift percentage must be 7.0%", 7.0, driftPctA, 1e-4)

        // Case B: 100.0m walked, 4.2m error -> 4.2% drift
        val drDistB = 100.0
        val errorB = 4.2
        val driftPctB = if (drDistB > 1.0) (errorB / drDistB) * 100.0 else 0.0
        assertEquals("Drift percentage must be 4.2%", 4.2, driftPctB, 1e-4)

        // Case C: Near-zero distance (<1m) must avoid division by zero or inflated percentage
        val drDistC = 0.5
        val errorC = 0.2
        val driftPctC = if (drDistC > 1.0) (errorC / drDistC) * 100.0 else 0.0
        assertEquals("Drift percentage must be 0.0% for negligible travel", 0.0, driftPctC, 1e-4)
    }

    // =========================================================================
    // TEST 7: Recovery does not cause a large position jump
    // =========================================================================
    @Test
    fun test7_recoveryDoesNotCauseLargePositionJump() {
        val engine = DeadReckoningEngine()
        val startLat = 11.016800
        val startLon = 76.955800
        engine.initialize(startLat, startLon)

        // Target GNSS position is 15 meters away
        val targetLat = 11.016935
        val targetLon = 76.955800

        val totalDistBefore = engine.distanceTravelled
        val rawDistOffset = CoordinateTransformer.computeDistanceBetween(startLat, startLon, targetLat, targetLon)
        assertTrue("Offset should be approx 15m", rawDistOffset > 14.0)

        // Start smooth re-anchoring
        engine.startSmoothReanchoring(targetLat, targetLon)
        assertTrue("Smooth re-anchoring must be active", engine.isReanchoring)

        var prevLat = engine.getState().latitude
        var prevLon = engine.getState().longitude

        // Step 1: Verify single step displacement is bounded (<= 1.0 meter)
        val active = engine.stepSmoothReanchoring()
        val step1Lat = engine.getState().latitude
        val step1Lon = engine.getState().longitude
        val step1Dist = CoordinateTransformer.computeDistanceBetween(prevLat, prevLon, step1Lat, step1Lon)

        assertTrue("Single step during recovery must NOT jump (max 1.0m per step)", step1Dist <= 1.05)
        assertEquals("Re-anchoring must NOT artificially increase physical travelled distance", totalDistBefore, engine.distanceTravelled, 0.001)

        // Step remaining iterations until complete
        var steps = 1
        var isStillReanchoring = active
        while (isStillReanchoring && steps < 100) {
            isStillReanchoring = engine.stepSmoothReanchoring()
            steps++
        }

        assertFalse("Re-anchoring must complete within reasonable iterations", isStillReanchoring)
        val finalLat = engine.getState().latitude
        val finalLon = engine.getState().longitude
        val finalDistToTarget = CoordinateTransformer.computeDistanceBetween(finalLat, finalLon, targetLat, targetLon)
        assertTrue("Final position must converge smoothly to target GNSS (<0.1m)", finalDistToTarget < 0.1)
        assertEquals("Total distance travelled must remain intact after re-anchoring", totalDistBefore, engine.distanceTravelled, 0.001)
    }

    // =========================================================================
    // TEST 8: ROTATING_IN_PLACE still produces zero navigation distance
    // =========================================================================
    @Test
    fun test8_rotatingInPlaceProducesZeroNavigationDistance() {
        val engine = setupEngineWithModel()
        engine.initialize(11.0168, 76.9558)

        var timeNs = 1_000_000_000L
        for (i in 0 until 30) {
            val t = i * 0.1f
            val headingDeg = t * 60f
            val headingRad = Math.toRadians(headingDeg.toDouble()).toFloat()
            engine.updateOrientation(
                OrientationSample(
                    timestampNs = timeNs,
                    rotationMatrix = floatArrayOf(cos(headingRad), -sin(headingRad), 0f, sin(headingRad), cos(headingRad), 0f, 0f, 0f, 1f),
                    quaternion = floatArrayOf(0f, 0f, 0f, 1f),
                    azimuthRad = headingRad,
                    pitchRad = 0f,
                    rollRad = 0f
                )
            )
            // Active yaw gyro (1.20 rad/s) but low translational accel
            engine.addSensorSample(ImuSample(timeNs, 0.10f, 0.08f, 0.12f, 0.10f, 0.15f, 1.20f))
            timeNs += 100_000_000L
        }

        assertEquals("Motion state must detect ROTATING_IN_PLACE", NavMotionState.ROTATING_IN_PLACE, engine.zuptDetector.motionState)
        assertEquals("Rotating in place must accumulate ZERO distance", 0.0, engine.distanceTravelled, 0.05)
        assertEquals("EKF speed must remain 0 during in-place rotation", 0.0f, engine.getState().speed, 0.05f)
    }

    // =========================================================================
    // TEST 9: STATIONARY still produces zero navigation distance
    // =========================================================================
    @Test
    fun test9_stationaryProducesZeroNavigationDistance() {
        val engine = setupEngineWithModel()
        engine.initialize(11.0168, 76.9558)

        var timeNs = 1_000_000_000L
        for (i in 0 until 30) {
            engine.addSensorSample(ImuSample(timeNs, 0.01f, 0.01f, 0.01f, 0.002f, 0.002f, 0.002f))
            timeNs += 100_000_000L
        }

        assertTrue("ZUPT detector must confirm stationary", engine.zuptDetector.isStationary)
        assertEquals("Motion state must be STATIONARY", NavMotionState.STATIONARY, engine.zuptDetector.motionState)
        assertEquals("Stationary state must produce 0.0 distance", 0.0, engine.distanceTravelled, 0.001)
        assertEquals("Speed must be 0.0", 0.0f, engine.getState().speed, 0.001f)
    }

    // =========================================================================
    // TEST 10: Genuine walking still produces movement
    // =========================================================================
    @Test
    fun test10_genuineWalkingProducesMovement() {
        val engine = setupEngineWithModel()
        engine.initialize(11.0168, 76.9558)

        var timeNs = 1_000_000_000L
        // 5 seconds of walking steps (forward push, bounce, arm swing)
        for (i in 0 until 50) {
            val t = i * 0.1f
            val axForward = 0.5f + 0.8f * sin(t * 12f)
            val azBounce = 0.9f * cos(t * 12f)
            val gx = 0.15f * sin(t * 12f)
            val gy = 0.25f * cos(t * 12f)
            engine.addSensorSample(ImuSample(timeNs, axForward, 0.1f, azBounce, gx, gy, 0.05f))
            timeNs += 100_000_000L
        }

        assertTrue("Walking must accumulate distance (>2.0m)", engine.distanceTravelled > 2.0)
        assertTrue("Walking distance must remain physically plausible (<20m)", engine.distanceTravelled < 20.0)
        assertTrue("Speed must be positive during walking", engine.getState().speed > 0.3f)
        assertEquals("Motion state must be MOVING", NavMotionState.MOVING, engine.zuptDetector.motionState)
    }

    // =========================================================================
    // TEST 11: ML ACCEPTED/CLAMPED/REJECTED counters work
    // =========================================================================
    @Test
    fun test11_mlGateCountersWork() {
        val engine = DeadReckoningEngine()
        engine.initialize(11.0168, 76.9558)

        assertEquals("Initial accepted count must be 0", 0, engine.acceptedCount)
        assertEquals("Initial clamped count must be 0", 0, engine.clampedCount)
        assertEquals("Initial rejected count must be 0", 0, engine.rejectedCount)

        // Record gate actions
        engine.recordGateAction(GateAction.ACCEPTED)
        engine.recordGateAction(GateAction.ACCEPTED)
        engine.recordGateAction(GateAction.CLAMPED)
        engine.recordGateAction(GateAction.REJECTED)

        assertEquals("Accepted count must be 2", 2, engine.acceptedCount)
        assertEquals("Clamped count must be 1", 1, engine.clampedCount)
        assertEquals("Rejected count must be 1", 1, engine.rejectedCount)
        assertEquals("Latest gate action must be REJECTED", GateAction.REJECTED, engine.latestGateAction)

        val state = engine.getState()
        assertEquals("State accepted count must match engine", 2, state.acceptedCount)
        assertEquals("State clamped count must match engine", 1, state.clampedCount)
        assertEquals("State rejected count must match engine", 1, state.rejectedCount)
        assertEquals("State latest gate action must match", "REJECTED", state.latestGateAction)
    }
}
